from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
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
    SUMMARY_SYSTEM_PROMPT,
)
from graphtool.agents.knowledge.state import (
    AgentResponse,
    AgentState,
    ConversationSummary,
    EvidenceRecord,
    FinalAnswerDraft,
    ResearchDecision,
    ResearchQuery,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.tools import search_knowledge_base
from graphtool.agents.knowledge.workflow_context import (
    answer_text,
    conversation_token_count,
    evaluation_text,
    merge_references,
    research_context,
    summary_text,
    unique_ordered,
)
from graphtool.runtime import GraphToolRuntime

MAX_SEARCHES_PER_TURN = 5
NO_EVIDENCE_DISCLOSURE = (
    "I couldn't find supporting information in the knowledge base. The following "
    "is a best-effort answer based on general knowledge and is not verified "
    "against the knowledge base."
)


def build_workflow_graph(
    model: BaseChatModel,
    runtime: GraphToolRuntime,
    checkpointer: InMemorySaver,
    *,
    compact_trigger_tokens: int,
    compact_recent_tokens: int,
):
    summary_model = model.with_structured_output(ConversationSummary)
    research_model = model.with_structured_output(ResearchDecision)
    refine_model = model.with_structured_output(ResearchQuery)
    evaluator_model = model.with_structured_output(SufficiencyDecision)
    answer_model = model.with_structured_output(FinalAnswerDraft)

    def compact(state: AgentState) -> dict:
        summary = state.get("conversation_summary", "")
        messages = state["messages"]
        if conversation_token_count(summary, messages) < compact_trigger_tokens:
            return {}

        retained_messages = trim_messages(
            messages,
            max_tokens=compact_recent_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            allow_partial=False,
            start_on="human",
        )
        if not retained_messages:
            retained_messages = [messages[-1]]
        retained_ids = {message.id for message in retained_messages}
        messages_to_summarize = [
            message for message in messages if message.id not in retained_ids
        ]
        if not messages_to_summarize:
            return {}

        updated_summary = _validated_output(
            ConversationSummary,
            summary_model.invoke(
                [
                    SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(
                        content=summary_text(summary, messages_to_summarize)
                    ),
                ]
            ),
        )
        return {
            "conversation_summary": updated_summary.summary.strip(),
            "messages": [
                RemoveMessage(id=message.id)
                for message in messages_to_summarize
                if message.id is not None
            ],
        }

    def research(state: AgentState) -> dict:
        decision = _validated_output(
            ResearchDecision,
            research_model.invoke(
                [
                    SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                    HumanMessage(content=research_context(state)),
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
                    SystemMessage(content=REFINE_SYSTEM_PROMPT),
                    HumanMessage(content=research_context(state)),
                ]
            ),
        )
        return {"proposed_query": query.query.strip(), "direct_response": None}

    def search(state: AgentState) -> dict:
        result = search_knowledge_base(runtime, state["proposed_query"] or "")
        references, reference_ids = merge_references(
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
                    HumanMessage(content=evaluation_text(state)),
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
        system_prompt = (
            NO_EVIDENCE_ANSWER_SYSTEM_PROMPT
            if no_evidence
            else ANSWER_SYSTEM_PROMPT
        )
        prompt_text = answer_text(state, partial=partial)
        references_by_id = {
            item.id: item.reference for item in state["references"]
        }
        draft = _validated_output(
            FinalAnswerDraft,
            answer_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt_text),
                ]
            ),
        )
        cited_ids = unique_ordered(draft.cited_reference_ids)
        unknown_ids = [
            reference_id
            for reference_id in cited_ids
            if reference_id not in references_by_id
        ]
        if unknown_ids:
            valid_ids = ", ".join(references_by_id) or "[None]"
            invalid_ids = ", ".join(unknown_ids)
            draft = _validated_output(
                FinalAnswerDraft,
                answer_model.invoke(
                    [
                        SystemMessage(
                            content=(
                                f"{system_prompt}\n\n"
                                "Your previous draft cited unknown reference IDs: "
                                f"{invalid_ids}. Regenerate the answer using only "
                                f"these available reference IDs: {valid_ids}. Remove "
                                "or qualify any claim that the available evidence "
                                "does not support."
                            )
                        ),
                        HumanMessage(content=prompt_text),
                    ]
                ),
            )
            cited_ids = unique_ordered(draft.cited_reference_ids)
            unknown_ids = [
                reference_id
                for reference_id in cited_ids
                if reference_id not in references_by_id
            ]
            if unknown_ids:
                joined = ", ".join(unknown_ids)
                raise RuntimeError(
                    "Knowledge agent answer cited unknown references after retry: "
                    f"{joined}."
                )
        cited_references = [
            references_by_id[reference_id] for reference_id in cited_ids
        ]
        if state["references"] and not cited_references:
            raise RuntimeError(
                "Knowledge agent answer did not cite retrieved evidence."
            )
        response_text = draft.answer.strip()
        if no_evidence:
            response_text = f"{NO_EVIDENCE_DISCLOSURE}\n\n{response_text}"
        response = AgentResponse(
            answer=response_text,
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

    def cleanup(state: AgentState) -> dict:
        return {
            "question": "",
            "evidence": [],
            "references": [],
            "search_count": 0,
            "research_action": None,
            "proposed_query": None,
            "direct_response": None,
            "evaluation": None,
        }

    builder = StateGraph(AgentState)
    builder.add_node("compact", compact)
    builder.add_node("research", research)
    builder.add_node("refine", refine)
    builder.add_node("search", search)
    builder.add_node("evaluate", evaluate)
    builder.add_node("answer", answer)
    builder.add_node("finish_conversation", finish_conversation)
    builder.add_node("cleanup", cleanup)
    builder.add_edge(START, "compact")
    builder.add_edge("compact", "research")
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
    builder.add_edge("answer", "cleanup")
    builder.add_edge("finish_conversation", "cleanup")
    builder.add_edge("cleanup", END)
    return builder.compile(checkpointer=checkpointer)


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


def _validated_output(model_type: type[BaseModel], value):
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)
