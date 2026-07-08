from collections.abc import Sequence
from datetime import datetime, timezone

from graphtool.graph.types import GraphMetadata, KnowledgeGraph
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage

SYSTEM_PROMPT = (
    "You extract knowledge graphs from text. Identify the key entities as nodes "
    "and the relationships between them as edges. Every edge must reference "
    "existing node ids. Return only the structured knowledge graph. "
    "Leave the metadata field empty."
)

USER_PROMPT_TEMPLATE = "Extract the knowledge graph from the following markdown:\n\n{markdown}"


def generate_knowledge_graph(
    markdown: str,
    source: str,
    llm: LLMClient,
) -> KnowledgeGraph:
    messages: Sequence[LLMMessage] = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=USER_PROMPT_TEMPLATE.format(markdown=markdown)),
    ]
    graph = llm.generate_structured(messages, KnowledgeGraph)
    graph.metadata = GraphMetadata(
        source=source,
        model=None,
        created_at=datetime.now(timezone.utc),
    )
    return graph