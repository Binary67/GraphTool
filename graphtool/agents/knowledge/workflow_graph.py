import logging
from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from openai import APITimeoutError
from pydantic import BaseModel

from graphtool.agents.knowledge.prompts import (
    ANSWER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)
from graphtool.agents.knowledge.state import (
    AgentResponse,
    AgentChunkReference,
    AgentState,
    ConversationSummary,
    EvidenceRecord,
    FinalAnswerDraft,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.tools import (
    ChunkNeighborhoodArtifact,
    KnowledgeSearchArtifact,
    ToolErrorArtifact,
    create_knowledge_tools,
)
from graphtool.agents.knowledge.workflow_context import (
    answer_text,
    conversation_token_count,
    evaluation_text,
    merge_references,
    research_context,
    summary_text,
    unique_ordered,
)
from graphtool.run_logging import LOGGER_NAME
from graphtool.runtime import GraphToolRuntime

MAX_RETRIEVALS_PER_TURN = 5
RUN_LOGGER = logging.getLogger(LOGGER_NAME)
RESEARCH_TOOL_CORRECTION = (
    "Your previous response did not call a retrieval tool. The available "
    "evidence is insufficient, so call exactly one retrieval tool now. Do not "
    "answer with prose."
)
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
    tools = create_knowledge_tools(runtime)
    summary_model = model.with_structured_output(ConversationSummary)
    research_model = model.bind_tools(tools, parallel_tool_calls=False)
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

        summary_messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(
                content=summary_text(summary, messages_to_summarize)
            ),
        ]
        summary_result, duration = _invoke_model(
            summary_model,
            summary_messages,
            stage="conversation summary",
        )
        updated_summary = _validated_output(
            ConversationSummary,
            summary_result,
        )
        RUN_LOGGER.info(
            "Conversation summary completed in %.2fs",
            duration,
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
        follow_up = (
            state.get("evaluation") is not None
            and state["evaluation"].verdict == "insufficient"
        )
        round_number = state["retrieval_count"] + 1
        research_messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=research_context(state)),
            *state["messages"],
        ]
        try:
            response, duration = _invoke_model(
                research_model,
                research_messages,
                stage=f"research round {round_number}",
            )
            research_duration = duration
            if not isinstance(response, AIMessage):
                raise TypeError(
                    "Tool-bound research model must return an AIMessage."
                )
            if follow_up and not response.tool_calls:
                RUN_LOGGER.info(
                    "Research round %d did not select a tool; retrying with a "
                    "correction",
                    round_number,
                )
                response, correction_duration = _invoke_model(
                    research_model,
                    [
                        *research_messages,
                        response,
                        HumanMessage(content=RESEARCH_TOOL_CORRECTION),
                    ],
                    stage=f"research round {round_number} corrective retry",
                )
                research_duration += correction_duration
                if not isinstance(response, AIMessage):
                    raise TypeError(
                        "Tool-bound research model must return an AIMessage."
                    )
        except APITimeoutError:
            if follow_up and state["retrieval_count"] > 0:
                RUN_LOGGER.warning(
                    "Follow-up research timed out; answering with the evidence "
                    "already retrieved"
                )
                return {"research_action": "answer", "direct_response": None}
            raise
        RUN_LOGGER.info(
            "Research round %d completed in %.2fs",
            round_number,
            research_duration,
        )
        if response.tool_calls:
            _log_tool_selection(response.tool_calls[0])
            return {
                "messages": [response],
                "research_action": "tools",
                "direct_response": None,
                "evaluation": None,
            }
        if follow_up:
            if state["retrieval_count"] == 0:
                raise RuntimeError(
                    "Research model did not select a retrieval tool after correction."
                )
            RUN_LOGGER.warning(
                "Follow-up research did not select a tool after correction; "
                "answering with the evidence already retrieved"
            )
            return {"research_action": "answer", "direct_response": None}
        return {
            "research_action": "respond",
            "direct_response": _message_text(response).strip(),
        }

    def record_tool_results(state: AgentState) -> dict:
        evidence = list(state["evidence"])
        references = list(state["references"])
        allowed_chunks = list(state["allowed_chunks"])
        used_neighborhoods = list(state["used_neighborhoods"])
        search_count = state["search_count"]
        retrieval_count = state["retrieval_count"]
        tool_messages = _trailing_tool_messages(state["messages"])

        for message in tool_messages:
            artifact = _tool_artifact(message)
            retrieval_count += 1
            if isinstance(artifact, KnowledgeSearchArtifact):
                references, reference_ids = merge_references(
                    references,
                    artifact.references,
                )
                evidence.append(
                    EvidenceRecord(
                        query=artifact.query,
                        context_text=artifact.context_text,
                        reference_ids=reference_ids,
                    )
                )
                allowed_chunks = _merge_allowed_chunks(
                    allowed_chunks,
                    artifact.chunks,
                )
                search_count += 1
            elif isinstance(artifact, ChunkNeighborhoodArtifact):
                references, reference_ids = merge_references(
                    references,
                    artifact.references,
                )
                evidence.append(
                    EvidenceRecord(
                        query=(
                            "Chunk neighborhood: "
                            f"{artifact.source} :: {artifact.chunk_id}"
                        ),
                        context_text=artifact.context_text,
                        reference_ids=reference_ids,
                    )
                )
                key = _chunk_key(artifact.source, artifact.chunk_id)
                if key not in used_neighborhoods:
                    used_neighborhoods.append(key)
            else:
                error = (
                    artifact.message
                    if isinstance(artifact, ToolErrorArtifact)
                    else str(message.content)
                )
                evidence.append(
                    EvidenceRecord(
                        query=f"Tool error: {message.name or 'unknown'}",
                        context_text=error,
                    )
                )
                if message.name == "search_knowledge_base":
                    search_count += 1

        tool_message_ids = list(state["tool_message_ids"])
        exchange_messages = _tool_exchange_messages(
            state["messages"],
            tool_messages,
        )
        for message in exchange_messages:
            if message.id is not None and message.id not in tool_message_ids:
                tool_message_ids.append(message.id)
        return {
            "evidence": evidence,
            "references": references,
            "allowed_chunks": allowed_chunks,
            "used_neighborhoods": used_neighborhoods,
            "search_count": search_count,
            "retrieval_count": retrieval_count,
            "tool_message_ids": tool_message_ids,
            "research_action": None,
        }

    def evaluate(state: AgentState) -> dict:
        round_number = state["retrieval_count"]
        evaluation_messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_text(state)),
        ]
        evaluation_result, duration = _invoke_model(
            evaluator_model,
            evaluation_messages,
            stage=f"evidence evaluation round {round_number}",
        )
        decision = _validated_output(
            SufficiencyDecision,
            evaluation_result,
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
        RUN_LOGGER.info(
            "Evidence evaluation round %d completed in %.2fs: %s",
            round_number,
            duration,
            decision.verdict,
        )
        if decision.missing_information:
            RUN_LOGGER.info(
                "Missing information: %s",
                decision.missing_information,
            )
        return {"evaluation": decision}

    def answer(state: AgentState) -> dict:
        answer_started_at = perf_counter()
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
        answer_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_text),
        ]
        answer_result, _ = _invoke_model(
            answer_model,
            answer_messages,
            stage="answer generation",
        )
        draft = _validated_output(
            FinalAnswerDraft,
            answer_result,
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
            retry_messages = [
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
            retry_result, _ = _invoke_model(
                answer_model,
                retry_messages,
                stage="answer citation retry",
            )
            draft = _validated_output(
                FinalAnswerDraft,
                retry_result,
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
        RUN_LOGGER.info(
            "Answer completed in %.2fs: status=%s, references=%d",
            perf_counter() - answer_started_at,
            response.status,
            len(response.references),
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
            "messages": [
                RemoveMessage(id=message_id)
                for message_id in state["tool_message_ids"]
            ],
            "question": "",
            "evidence": [],
            "references": [],
            "search_count": 0,
            "retrieval_count": 0,
            "allowed_chunks": [],
            "used_neighborhoods": [],
            "tool_message_ids": [],
            "research_action": None,
            "direct_response": None,
            "evaluation": None,
        }

    builder = StateGraph(AgentState)
    builder.add_node("compact", compact)
    builder.add_node("research", research)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("record_tool_results", record_tool_results)
    builder.add_node("evaluate", evaluate)
    builder.add_node("answer", answer)
    builder.add_node("finish_conversation", finish_conversation)
    builder.add_node("cleanup", cleanup)
    builder.add_edge(START, "compact")
    builder.add_edge("compact", "research")
    builder.add_conditional_edges(
        "research",
        _route_research,
        {"tools": "tools", "evaluate": "evaluate", "answer": "answer"},
    )
    builder.add_edge("tools", "record_tool_results")
    builder.add_edge("record_tool_results", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        _route_evaluation,
        {
            "answer": "answer",
            "finish_conversation": "finish_conversation",
            "research": "research",
        },
    )
    builder.add_edge("answer", "cleanup")
    builder.add_edge("finish_conversation", "cleanup")
    builder.add_edge("cleanup", END)
    return builder.compile(checkpointer=checkpointer)


def _route_research(state: AgentState) -> str:
    if state.get("research_action") == "tools":
        return "tools"
    if state.get("research_action") == "respond":
        return "evaluate"
    if state.get("research_action") == "answer":
        return "answer"
    raise RuntimeError("Research action is missing.")


def _route_evaluation(state: AgentState) -> str:
    evaluation = state["evaluation"]
    if evaluation is None:
        raise RuntimeError("Evidence evaluation is missing.")
    if evaluation.verdict == "conversation":
        return "finish_conversation"
    if (
        evaluation.verdict == "sufficient"
        or state["retrieval_count"] >= MAX_RETRIEVALS_PER_TURN
    ):
        return "answer"
    return "research"


def _log_tool_selection(tool_call: dict) -> None:
    name = str(tool_call.get("name", "unknown"))
    arguments = tool_call.get("args", {})
    RUN_LOGGER.info("Research selected: %s", name)
    if not isinstance(arguments, dict):
        return
    if name == "search_knowledge_base":
        RUN_LOGGER.info("Search query: %s", arguments.get("query", ""))
    elif name == "get_chunk_neighborhood":
        RUN_LOGGER.info(
            "Chunk neighborhood: %s :: %s",
            arguments.get("source", ""),
            arguments.get("chunk_id", ""),
        )


def _invoke_model(model, messages: list, *, stage: str):
    RUN_LOGGER.info(
        "Starting %s: prompt approximately %d tokens",
        stage,
        count_tokens_approximately(messages),
    )
    started_at = perf_counter()
    try:
        return model.invoke(messages), perf_counter() - started_at
    except Exception as exc:
        duration = perf_counter() - started_at
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            RUN_LOGGER.error(
                "%s failed after %.2fs: %s",
                stage.capitalize(),
                duration,
                type(exc).__name__,
            )
        else:
            RUN_LOGGER.error(
                "%s failed after %.2fs: %s (status=%s)",
                stage.capitalize(),
                duration,
                type(exc).__name__,
                status_code,
            )
        raise


def _validated_output(model_type: type[BaseModel], value):
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def _trailing_tool_messages(messages) -> list[ToolMessage]:
    trailing = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        trailing.append(message)
    return list(reversed(trailing))


def _tool_exchange_messages(messages, tool_messages: list[ToolMessage]):
    if not tool_messages:
        return []
    first_tool_index = len(messages) - len(tool_messages)
    preceding = messages[first_tool_index - 1] if first_tool_index > 0 else None
    if isinstance(preceding, AIMessage) and preceding.tool_calls:
        return [preceding, *tool_messages]
    return tool_messages


def _tool_artifact(
    message: ToolMessage,
) -> KnowledgeSearchArtifact | ChunkNeighborhoodArtifact | ToolErrorArtifact | None:
    artifact = message.artifact
    if isinstance(
        artifact,
        (KnowledgeSearchArtifact, ChunkNeighborhoodArtifact, ToolErrorArtifact),
    ):
        return artifact
    if not isinstance(artifact, dict):
        return None
    artifact_type = artifact.get("type")
    if artifact_type == "search":
        return KnowledgeSearchArtifact.model_validate(artifact)
    if artifact_type == "chunk_neighborhood":
        return ChunkNeighborhoodArtifact.model_validate(artifact)
    if artifact_type == "error":
        return ToolErrorArtifact.model_validate(artifact)
    return None


def _merge_allowed_chunks(
    existing: list[AgentChunkReference],
    incoming: list[AgentChunkReference],
) -> list[AgentChunkReference]:
    merged = list(existing)
    keys = {_chunk_key(item.source, item.chunk_id) for item in existing}
    for item in incoming:
        key = _chunk_key(item.source, item.chunk_id)
        if key not in keys:
            merged.append(item)
            keys.add(key)
    return merged


def _chunk_key(source: str, chunk_id: str) -> str:
    return f"{source} :: {chunk_id}"
