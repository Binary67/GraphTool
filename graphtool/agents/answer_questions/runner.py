import json
from collections.abc import Iterable
from typing import Any

from graphtool.agents.answer_questions.graph import build_answer_question_graph
from graphtool.agents.answer_questions.model import make_answer_chat_model
from graphtool.agents.answer_questions.tools import make_retrieve_knowledge_context_tool
from graphtool.agents.answer_questions.types import AnswerResult, RetrievedContext
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.runtime import create_runtime

MAX_AGENT_ITERATIONS = 6


def answer_question(
    question: str,
    config: AzureOpenAIConfig,
) -> AnswerResult:
    runtime = create_runtime(config)
    model = make_answer_chat_model(config)
    tool = make_retrieve_knowledge_context_tool(
        runtime.graph_store,
        runtime.chunk_store,
        knowledge_base_store=runtime.knowledge_base_store,
        embedding_client=runtime.fast_llm,
        chunk_embedding_store=runtime.chunk_embedding_store,
    )
    graph = build_answer_question_graph(model, [tool])
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": MAX_AGENT_ITERATIONS},
    )
    messages = list(result.get("messages", []))
    retrievals = _extract_retrievals(messages)
    return AnswerResult(
        question=question,
        answer=_message_text(messages[-1]) if messages else "",
        sources=_unique_ordered(
            source
            for retrieval in retrievals
            for source in retrieval.sources
        ),
        retrievals=retrievals,
    )


def _extract_retrievals(messages: list[Any]) -> list[RetrievedContext]:
    retrievals = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        content = _message_text(message)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue
        try:
            retrievals.append(RetrievedContext.model_validate(data))
        except ValueError:
            continue
    return retrievals


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_block_text(block) for block in content)
    return str(content)


def _content_block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        value = block.get("text") or block.get("content") or ""
        return value if isinstance(value, str) else str(value)
    return str(block)


def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
