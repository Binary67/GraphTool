from collections.abc import Sequence

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from graphtool.agents.knowledge.state import (
    AgentState,
    EvidenceReference,
)
from graphtool.retrieval import SourceReference, format_source_reference


def research_context(state: AgentState) -> str:
    prior_queries = [record.query for record in state["evidence"]]
    missing_information = (
        state["evaluation"].missing_information
        if state.get("evaluation") is not None
        else ""
    )
    return (
        "Conversation summary (context only, not evidence):\n"
        f"{state.get('conversation_summary') or '[None]'}\n\n"
        f"Original question: {state['question']}\n"
        f"Prior search queries: {prior_queries or ['None']}\n"
        f"Unresolved information: {missing_information or '[Not evaluated yet]'}"
    )


def evaluation_text(state: AgentState) -> str:
    return (
        f"Question:\n{state['question']}\n\n"
        f"Conversation:\n{_conversation_context_text(state)}\n\n"
        "Proposed conversational response:\n"
        f"{state.get('direct_response') or '[None]'}\n\n"
        f"Retrieved evidence:\n{_evidence_text(state)}"
    )


def answer_text(state: AgentState, *, partial: bool) -> str:
    missing_information = (
        state["evaluation"].missing_information
        if state.get("evaluation") is not None
        else ""
    )
    return (
        f"Question:\n{state['question']}\n\n"
        f"Conversation:\n{_conversation_context_text(state)}\n\n"
        f"Answer status: {'partial' if partial else 'complete'}\n"
        f"Unresolved information: {missing_information or '[None]'}\n\n"
        f"Retrieved evidence:\n{_evidence_text(state)}"
    )


def conversation_token_count(
    summary: str,
    messages: Sequence[AnyMessage],
) -> int:
    summary_messages = [HumanMessage(content=summary)] if summary else []
    return count_tokens_approximately([*summary_messages, *messages])


def summary_text(summary: str, messages: Sequence[AnyMessage]) -> str:
    return (
        f"Prior summary:\n{summary or '[None]'}\n\n"
        "Older messages to incorporate:\n"
        f"{_conversation_text(messages)}"
    )


def merge_references(
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
            merged.append(EvidenceReference(id=reference_id, reference=reference))
            ids_by_key[key] = reference_id
        result_ids.append(reference_id)
    return merged, unique_ordered(result_ids)


def unique_ordered(values: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


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


def _conversation_context_text(state: AgentState) -> str:
    summary = state.get("conversation_summary") or "[None]"
    return (
        f"Summary (context only, not evidence):\n{summary}\n\n"
        f"Recent messages:\n{_conversation_text(state['messages'])}"
    )


def _conversation_text(messages: Sequence[AnyMessage]) -> str:
    return "\n".join(
        f"{message.type}: {_message_text(message)}" for message in messages
    )


def _message_text(message: AnyMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def _reference_key(
    reference: SourceReference,
) -> tuple[str, int | None, int | None]:
    return reference.source, reference.page_start, reference.page_end


def _format_reference(reference: SourceReference) -> str:
    return format_source_reference(reference)
