from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from graphtool.agents.knowledge.prompts import (
    ANSWER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from graphtool.agents.knowledge.state import (
    AgentResponse,
    AgentState,
    EvidenceRecord,
    EvidenceReference,
    FinalAnswerDraft,
    ResearchDecision,
    ResearchQuery,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.tools import search_knowledge_base
from graphtool.retrieval import SourceReference
from graphtool.runtime import GraphToolRuntime

MAX_SEARCHES_PER_TURN = 5
NO_EVIDENCE_DISCLOSURE = (
    "I couldn't find supporting information in the knowledge base. The following "
    "is a best-effort answer based on general knowledge and is not verified "
    "against the knowledge base."
)


class KnowledgeAgent:
    def __init__(
        self,
        model: BaseChatModel,
        runtime: GraphToolRuntime,
    ) -> None:
        self._runtime = runtime
        self._graph = _build_graph(model, runtime)

    def ask(self, question: str, *, thread_id: str) -> AgentResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("Thread ID must not be empty.")
        if not self._runtime.knowledge_base_store.exists():
            raise FileNotFoundError(
                "Knowledge base not found. Synchronize documents before asking."
            )

        result = self._graph.invoke(
            {
                "messages": [HumanMessage(content=normalized_question)],
                "question": normalized_question,
                "evidence": [],
                "references": [],
                "search_count": 0,
                "research_action": None,
                "proposed_query": None,
                "direct_response": None,
                "evaluation": None,
                "response": None,
            },
            config={
                "configurable": {"thread_id": normalized_thread_id},
                "recursion_limit": 50,
            },
        )
        response = result.get("response")
        if not isinstance(response, AgentResponse):
            raise RuntimeError("Knowledge agent completed without a response.")
        return response


def create_knowledge_agent(
    model: BaseChatModel,
    runtime: GraphToolRuntime,
) -> KnowledgeAgent:
    return KnowledgeAgent(model, runtime)


def _build_graph(model: BaseChatModel, runtime: GraphToolRuntime):
    research_model = model.with_structured_output(ResearchDecision)
    refine_model = model.with_structured_output(ResearchQuery)
    evaluator_model = model.with_structured_output(SufficiencyDecision)
    answer_model = model.with_structured_output(FinalAnswerDraft)

    def research(state: AgentState) -> dict:
        decision = _validated_output(
            ResearchDecision,
            research_model.invoke(
                [
                    SystemMessage(
                        content=_research_prompt(RESEARCH_SYSTEM_PROMPT, state)
                    ),
                    *state["messages"],
                ]
            ),
        )
        return {
            "research_action": decision.action,
            "proposed_query": (
                decision.query.strip() if decision.query is not None else None
            ),
            "direct_response": (
                decision.response.strip() if decision.response is not None else None
            ),
        }

    def refine(state: AgentState) -> dict:
        query = _validated_output(
            ResearchQuery,
            refine_model.invoke(
                [
                    SystemMessage(content=_research_prompt(REFINE_SYSTEM_PROMPT, state)),
                    HumanMessage(content=state["question"]),
                ]
            ),
        )
        return {
            "proposed_query": query.query.strip(),
            "direct_response": None,
        }

    def search(state: AgentState) -> dict:
        result = search_knowledge_base(runtime, state["proposed_query"] or "")
        references, reference_ids = _merge_references(
            state["references"],
            result.references,
        )
        evidence = EvidenceRecord(
            query=result.query,
            context_text=result.context_text,
            reference_ids=reference_ids,
        )
        return {
            "evidence": [*state["evidence"], evidence],
            "references": references,
            "search_count": state["search_count"] + 1,
            "proposed_query": None,
        }

    def evaluate(state: AgentState) -> dict:
        decision = _validated_output(
            SufficiencyDecision,
            evaluator_model.invoke(
                [
                    SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
                    HumanMessage(content=_evaluation_text(state)),
                ]
            ),
        )
        if decision.verdict == "sufficient" and not state["references"]:
            decision = SufficiencyDecision(
                verdict="insufficient",
                missing_information=(
                    decision.missing_information
                    or "No knowledge-base evidence has been retrieved."
                ),
            )
        if decision.verdict == "conversation" and (
            state["evidence"] or not state.get("direct_response")
        ):
            decision = SufficiencyDecision(
                verdict="insufficient",
                missing_information=(
                    decision.missing_information
                    or "The request requires a knowledge-base-grounded answer."
                ),
            )
        return {"evaluation": decision}

    def answer(state: AgentState) -> dict:
        partial = (
            state["evaluation"] is None
            or state["evaluation"].verdict != "sufficient"
        )
        no_evidence = partial and not state["references"]
        draft = _validated_output(
            FinalAnswerDraft,
            answer_model.invoke(
                [
                    SystemMessage(
                        content=(
                            NO_EVIDENCE_ANSWER_SYSTEM_PROMPT
                            if no_evidence
                            else ANSWER_SYSTEM_PROMPT
                        )
                    ),
                    HumanMessage(content=_answer_text(state, partial=partial)),
                ]
            ),
        )
        references_by_id = {
            item.id: item.reference for item in state["references"]
        }
        cited_ids = _unique_ordered(draft.cited_reference_ids)
        cited_references = [
            references_by_id[reference_id]
            for reference_id in cited_ids
            if reference_id in references_by_id
        ]
        if state["references"] and not cited_references:
            raise RuntimeError(
                "Knowledge agent answer did not cite retrieved evidence."
            )
        answer_text = draft.answer.strip()
        if no_evidence:
            answer_text = f"{NO_EVIDENCE_DISCLOSURE}\n\n{answer_text}"
        response = AgentResponse(
            answer=answer_text,
            status="partial" if partial else "complete",
            references=cited_references,
            search_count=state["search_count"],
        )
        return {
            "messages": [AIMessage(content=response.answer)],
            "response": response,
        }

    def finish_conversation(state: AgentState) -> dict:
        response = AgentResponse(
            answer=state["direct_response"] or "",
            status="complete",
            references=[],
            search_count=state["search_count"],
        )
        return {
            "messages": [AIMessage(content=response.answer)],
            "response": response,
        }

    builder = StateGraph(AgentState)
    builder.add_node("research", research)
    builder.add_node("refine", refine)
    builder.add_node("search", search)
    builder.add_node("evaluate", evaluate)
    builder.add_node("answer", answer)
    builder.add_node("finish_conversation", finish_conversation)
    builder.add_edge(START, "research")
    builder.add_conditional_edges(
        "research",
        _route_research,
        {"search": "search", "evaluate": "evaluate"},
    )
    builder.add_edge("search", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        _route_evaluation,
        {
            "answer": "answer",
            "finish_conversation": "finish_conversation",
            "refine": "refine",
        },
    )
    builder.add_edge("refine", "search")
    builder.add_edge("answer", END)
    builder.add_edge("finish_conversation", END)
    return builder.compile(checkpointer=InMemorySaver())


def _route_research(state: AgentState) -> str:
    if state.get("research_action") == "search":
        return "search"
    if state.get("research_action") == "respond":
        return "evaluate"
    raise RuntimeError("Research action is missing.")


def _route_evaluation(state: AgentState) -> str:
    evaluation = state["evaluation"]
    if evaluation is None:
        raise RuntimeError("Evidence evaluation is missing.")
    if evaluation.verdict == "conversation":
        return "finish_conversation"
    if (
        evaluation.verdict == "sufficient"
        or state["search_count"] >= MAX_SEARCHES_PER_TURN
    ):
        return "answer"
    return "refine"


def _research_prompt(prompt: str, state: AgentState) -> str:
    prior_queries = [record.query for record in state["evidence"]]
    missing_information = (
        state["evaluation"].missing_information
        if state.get("evaluation") is not None
        else ""
    )
    return (
        f"{prompt}\n"
        f"Original question: {state['question']}\n"
        f"Prior search queries: {prior_queries or ['None']}\n"
        f"Unresolved information: {missing_information or '[Not evaluated yet]'}"
    )


def _evaluation_text(state: AgentState) -> str:
    return (
        f"Question:\n{state['question']}\n\n"
        f"Conversation:\n{_conversation_text(state['messages'])}\n\n"
        f"Proposed conversational response:\n"
        f"{state.get('direct_response') or '[None]'}\n\n"
        f"Retrieved evidence:\n{_evidence_text(state)}"
    )


def _answer_text(state: AgentState, *, partial: bool) -> str:
    missing_information = (
        state["evaluation"].missing_information
        if state.get("evaluation") is not None
        else ""
    )
    return (
        f"Question:\n{state['question']}\n\n"
        f"Conversation:\n{_conversation_text(state['messages'])}\n\n"
        f"Answer status: {'partial' if partial else 'complete'}\n"
        f"Unresolved information: {missing_information or '[None]'}\n\n"
        f"Retrieved evidence:\n{_evidence_text(state)}"
    )


def _evidence_text(state: AgentState) -> str:
    if not state["evidence"]:
        return "[None]"

    references = "\n".join(
        f"[{item.id}] {_format_reference(item.reference)}"
        for item in state["references"]
    )
    searches = "\n\n".join(
        (
            f"Search query: {record.query}\n"
            f"Available reference IDs: {record.reference_ids or ['None']}\n"
            f"{record.context_text}"
        )
        for record in state["evidence"]
    )
    return f"Reference registry:\n{references or '[None]'}\n\n{searches}"


def _conversation_text(messages: Sequence[AnyMessage]) -> str:
    return "\n".join(
        f"{message.type}: {_message_text(message)}" for message in messages
    )


def _message_text(message: AnyMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def _merge_references(
    existing: list[EvidenceReference],
    incoming: list[SourceReference],
) -> tuple[list[EvidenceReference], list[str]]:
    merged = list(existing)
    ids_by_key = {
        _reference_key(item.reference): item.id for item in existing
    }
    result_ids = []
    for reference in incoming:
        key = _reference_key(reference)
        reference_id = ids_by_key.get(key)
        if reference_id is None:
            reference_id = f"S{len(merged) + 1}"
            merged.append(
                EvidenceReference(id=reference_id, reference=reference)
            )
            ids_by_key[key] = reference_id
        result_ids.append(reference_id)
    return merged, _unique_ordered(result_ids)


def _reference_key(reference: SourceReference) -> tuple[str, int | None, int | None]:
    return reference.source, reference.page_start, reference.page_end


def _format_reference(reference: SourceReference) -> str:
    if reference.page_start is None:
        return reference.source
    if reference.page_start == reference.page_end:
        return f"{reference.source} (p. {reference.page_start})"
    return f"{reference.source} (pp. {reference.page_start}-{reference.page_end})"


def _unique_ordered(values: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _validated_output(model_type: type[BaseModel], value):
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)
