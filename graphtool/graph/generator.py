from collections.abc import Sequence
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage

SYSTEM_PROMPT = (
    "You extract knowledge graphs from text. Identify the key entities as nodes "
    "and the relationships between them as edges. Every edge must reference "
    "existing node ids. Return only the structured nodes and edges."
)

USER_PROMPT_TEMPLATE = (
    "Extract the knowledge graph from this markdown chunk.\n\n"
    "Chunk ID: {chunk_id}\n"
    "Source: {source}\n"
    "Heading path: {heading_path}\n\n"
    "{markdown}"
)


class _ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    type: str


class _ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    label: str


class _ExtractedKnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[_ExtractedNode]
    edges: list[_ExtractedEdge]


def generate_knowledge_graph(
    chunks: Sequence[Chunk],
    source: str,
    llm: LLMClient,
) -> KnowledgeGraph:
    graphs = [_generate_chunk_graph(chunk, llm) for chunk in chunks]
    graph = combine_knowledge_graphs(graphs)
    return graph.model_copy(
        update={
            "metadata": GraphMetadata(
                source=source,
                model=None,
                created_at=datetime.now(timezone.utc),
            )
        }
    )


def _generate_chunk_graph(chunk: Chunk, llm: LLMClient) -> KnowledgeGraph:
    messages: Sequence[LLMMessage] = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=USER_PROMPT_TEMPLATE.format(
                chunk_id=chunk.id,
                source=chunk.source,
                heading_path=" > ".join(chunk.heading_path),
                markdown=chunk.text,
            ),
        ),
    ]
    graph = llm.generate_structured(messages, _ExtractedKnowledgeGraph)
    return KnowledgeGraph(
        nodes=[
            Node(
                id=node.id,
                label=node.label,
                type=node.type,
                chunk_ids=[chunk.id],
            )
            for node in graph.nodes
        ],
        edges=[
            Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                label=edge.label,
                chunk_ids=[chunk.id],
            )
            for edge in graph.edges
        ],
    )


def combine_knowledge_graphs(graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
    nodes_by_id: dict[str, Node] = {}
    edges_by_key: dict[tuple[str, str, str], Edge] = {}

    for graph in graphs:
        for node in graph.nodes:
            existing_node = nodes_by_id.get(node.id)
            if existing_node is None:
                nodes_by_id[node.id] = node
                continue

            nodes_by_id[node.id] = existing_node.model_copy(
                update={
                    "chunk_ids": _extend_unique(existing_node.chunk_ids, node.chunk_ids)
                }
            )

        for edge in graph.edges:
            key = (edge.source, edge.target, edge.label)
            existing_edge = edges_by_key.get(key)
            if existing_edge is None:
                edges_by_key[key] = edge
                continue

            edges_by_key[key] = existing_edge.model_copy(
                update={
                    "chunk_ids": _extend_unique(existing_edge.chunk_ids, edge.chunk_ids)
                }
            )

    edges = [
        edge.model_copy(update={"id": f"edge-{index:04d}"})
        for index, edge in enumerate(edges_by_key.values(), start=1)
    ]
    return KnowledgeGraph(nodes=list(nodes_by_id.values()), edges=edges)


def _extend_unique(values: list[str], additions: list[str]) -> list[str]:
    merged = list(values)
    for addition in additions:
        if addition not in merged:
            merged.append(addition)
    return merged
