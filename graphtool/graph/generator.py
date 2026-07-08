import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


class GraphResolver(Protocol):
    def combine(self, graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
        ...

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
    *,
    resolver: GraphResolver | None = None,
    dropped_edges_path: Path | None = None,
) -> KnowledgeGraph:
    graphs = [
        _generate_chunk_graph(
            chunk,
            llm,
            dropped_edges_path=dropped_edges_path,
        )
        for chunk in chunks
    ]
    graph = (
        resolver.combine(graphs)
        if resolver is not None
        else combine_knowledge_graphs(graphs)
    )
    return graph.model_copy(
        update={
            "metadata": GraphMetadata(
                source=source,
                model=None,
                created_at=datetime.now(timezone.utc),
            )
        }
    )


def _generate_chunk_graph(
    chunk: Chunk,
    llm: LLMClient,
    *,
    dropped_edges_path: Path | None = None,
) -> KnowledgeGraph:
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
    nodes = [
        Node(
            id=node.id,
            label=node.label,
            type=node.type,
            chunk_ids=[chunk.id],
        )
        for node in graph.nodes
    ]
    node_ids = {node.id for node in nodes}
    edges = []
    for edge in graph.edges:
        missing = []
        if edge.source not in node_ids:
            missing.append("source")
        if edge.target not in node_ids:
            missing.append("target")
        if missing:
            _record_dropped_edge(chunk, edge, missing, dropped_edges_path)
            continue

        edges.append(
            Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                label=edge.label,
                chunk_ids=[chunk.id],
            )
        )

    return KnowledgeGraph(nodes=nodes, edges=edges)


def _record_dropped_edge(
    chunk: Chunk,
    edge: _ExtractedEdge,
    missing: list[str],
    dropped_edges_path: Path | None,
) -> None:
    RUN_LOGGER.warning(
        "Skipped extracted edge %s in %s: missing %s",
        edge.id,
        chunk.id,
        _missing_edge_description(edge, missing),
    )
    if dropped_edges_path is None:
        return

    dropped_edges_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": chunk.source,
        "chunk_id": chunk.id,
        "edge_id": edge.id,
        "label": edge.label,
        "edge_source": edge.source,
        "edge_target": edge.target,
        "missing": missing,
    }
    with dropped_edges_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True))
        file.write("\n")


def _missing_edge_description(edge: _ExtractedEdge, missing: list[str]) -> str:
    parts = []
    if "source" in missing:
        parts.append(f"source node {edge.source}")
    if "target" in missing:
        parts.append(f"target node {edge.target}")
    return ", ".join(parts)


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
                    "aliases": _merge_aliases(existing_node, node),
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


def _merge_aliases(existing: Node, incoming: Node) -> list[str]:
    additions = []
    if incoming.label != existing.label:
        additions.append(incoming.label)
    additions.extend(incoming.aliases)
    return _extend_unique(existing.aliases, additions)
