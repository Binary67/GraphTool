import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from graphtool.chunking.types import Chunk
from graphtool.graph.taxonomy import (
    CanonicalNodeType,
    TaxonomySuggestionStore,
    UNCLASSIFIED_NODE_TYPE,
    canonical_node_type_text,
    make_taxonomy_suggestion_records,
    normalize_type_name,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


class GraphResolver(Protocol):
    def combine(self, graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
        ...

    def combine_into(
        self,
        existing: KnowledgeGraph | None,
        graphs: Sequence[KnowledgeGraph],
    ) -> KnowledgeGraph:
        ...

SYSTEM_PROMPT = (
    "You extract compact knowledge graphs from markdown content. Identify only "
    "important domain entities as nodes and meaningful relationships as edges. "
    "Do not create nodes for prompt metadata, chunks, source file paths, "
    "markdown headings, tables, rows, columns, URLs, or generic document "
    "structure unless the content is explicitly about those concepts. Table "
    "contents can contain useful facts; extract those facts, not the table "
    "mechanics. Prefer a small graph of the most salient entities. Every edge "
    "must reference existing node ids. Node type must be one of: "
    f"{canonical_node_type_text()}. If none of those types fit, use "
    f"{UNCLASSIFIED_NODE_TYPE} and provide suggested_type with the missing "
    "taxonomy type. Return only the structured nodes and edges."
)

USER_PROMPT_TEMPLATE = (
    "Extract a compact knowledge graph from the markdown content below.\n\n"
    "Context only, do not extract this as graph content:\n"
    "Heading path: {heading_path}\n\n"
    "Markdown content:\n"
    "{markdown}"
)

_STRUCTURAL_NODE_TYPES = {
    "chunk",
    "column",
    "file path",
    "heading",
    "link",
    "markdown table",
    "row",
    "source",
    "source path",
    "table",
    "url",
}

_STRUCTURAL_NODE_LABELS = {
    "chunk",
    "column",
    "document",
    "file path",
    "heading",
    "markdown table",
    "row",
    "source",
    "source path",
    "table",
}


@dataclass(frozen=True)
class _GeneratedChunkGraph:
    graph: KnowledgeGraph
    raw_nodes: int
    kept_nodes: int
    dropped_structural_nodes: int
    raw_edges: int
    kept_edges: int


class _ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    type: CanonicalNodeType
    suggested_type: str | None = None

    @model_validator(mode="after")
    def validate_suggested_type(self) -> "_ExtractedNode":
        if self.type == UNCLASSIFIED_NODE_TYPE and not self.suggested_type:
            raise ValueError("suggested_type is required for unclassified nodes")
        if self.suggested_type is not None and not self.suggested_type.strip():
            raise ValueError("suggested_type cannot be blank")
        return self


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
    taxonomy_suggestion_store: TaxonomySuggestionStore | None = None,
) -> KnowledgeGraph:
    generated_chunks = []
    taxonomy_suggestion_records = []
    for chunk in chunks:
        generated = _generate_chunk_graph(
            chunk,
            llm,
            dropped_edges_path=dropped_edges_path,
        )
        generated_chunks.append(generated)
        if taxonomy_suggestion_store is not None:
            taxonomy_suggestion_records.extend(
                make_taxonomy_suggestion_records(
                    nodes=generated.graph.nodes,
                    source=chunk.source,
                    chunk_id=chunk.id,
                    model=_llm_model(llm),
                )
            )

    if taxonomy_suggestion_store is not None and taxonomy_suggestion_records:
        taxonomy_suggestion_store.append_many(taxonomy_suggestion_records)

    graphs = [generated.graph for generated in generated_chunks]
    graph = (
        resolver.combine(graphs)
        if resolver is not None
        else combine_knowledge_graphs(graphs)
    )
    _log_document_graph(source, len(chunks), generated_chunks, graph)
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
) -> _GeneratedChunkGraph:
    messages: Sequence[LLMMessage] = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=USER_PROMPT_TEMPLATE.format(
                heading_path=_heading_path_text(chunk),
                markdown=chunk.text,
            ),
        ),
    ]
    graph = llm.generate_structured(messages, _ExtractedKnowledgeGraph)
    structural_node_ids = _structural_node_ids(graph.nodes, chunk)
    nodes = _dedupe_chunk_nodes(graph.nodes, structural_node_ids, chunk)
    node_ids = {node.id for node in nodes}
    edges_by_key: dict[tuple[str, str, str], Edge] = {}
    for edge in graph.edges:
        if edge.source in structural_node_ids or edge.target in structural_node_ids:
            continue

        missing = []
        if edge.source not in node_ids:
            missing.append("source")
        if edge.target not in node_ids:
            missing.append("target")
        if missing:
            _record_dropped_edge(chunk, edge, missing, dropped_edges_path)
            continue

        key = (edge.source, edge.target, edge.label)
        edges_by_key.setdefault(
            key,
            Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                label=edge.label,
                chunk_ids=[chunk.id],
            ),
        )

    edges = [
        edge.model_copy(update={"id": f"edge-{index:04d}"})
        for index, edge in enumerate(edges_by_key.values(), start=1)
    ]
    generated = _GeneratedChunkGraph(
        graph=KnowledgeGraph(nodes=nodes, edges=edges),
        raw_nodes=len(graph.nodes),
        kept_nodes=len(nodes),
        dropped_structural_nodes=len(structural_node_ids),
        raw_edges=len(graph.edges),
        kept_edges=len(edges),
    )
    _log_chunk_graph(chunk, generated)
    return generated


def _dedupe_chunk_nodes(
    extracted_nodes: Sequence[_ExtractedNode],
    structural_node_ids: set[str],
    chunk: Chunk,
) -> list[Node]:
    nodes_by_id: dict[str, Node] = {}

    for node in extracted_nodes:
        if node.id in structural_node_ids:
            continue

        incoming = Node(
            id=node.id,
            label=node.label,
            type=node.type,
            suggested_type=node.suggested_type,
            chunk_ids=[chunk.id],
        )
        existing = nodes_by_id.get(incoming.id)
        if existing is None:
            nodes_by_id[incoming.id] = incoming
            continue

        nodes_by_id[incoming.id] = existing.model_copy(
            update={
                "type": _merge_node_type(existing, incoming),
                "aliases": _merge_aliases(existing, incoming),
                "suggested_type": existing.suggested_type or incoming.suggested_type,
            }
        )

    return list(nodes_by_id.values())


def _heading_path_text(chunk: Chunk) -> str:
    if not chunk.heading_path:
        return "(none)"
    return " > ".join(chunk.heading_path)


def _llm_model(llm: LLMClient) -> str | None:
    return getattr(llm, "model", None) or getattr(llm, "text_model", None)


def _structural_node_ids(nodes: Sequence[_ExtractedNode], chunk: Chunk) -> set[str]:
    return {
        node.id
        for node in nodes
        if _is_structural_node(node, chunk)
    }


def _is_structural_node(node: _ExtractedNode, chunk: Chunk) -> bool:
    normalized_type = _normalize_structural_text(node.type)
    if normalized_type in _STRUCTURAL_NODE_TYPES:
        return True

    normalized_label = _normalize_structural_text(node.label)
    if normalized_label in _STRUCTURAL_NODE_LABELS:
        return True

    metadata_values = _source_metadata_values(chunk)
    normalized_id = _normalize_structural_text(node.id)
    if normalized_id in metadata_values or normalized_label in metadata_values:
        return True

    return _is_chunk_wrapper(normalized_label, _heading_metadata_values(chunk))


def _source_metadata_values(chunk: Chunk) -> set[str]:
    values = {
        chunk.id,
        chunk.source,
    }
    return {
        normalized
        for normalized in (_normalize_structural_text(value) for value in values)
        if normalized
    }


def _heading_metadata_values(chunk: Chunk) -> set[str]:
    values = {
        _heading_path_text(chunk),
        *chunk.heading_path,
    }
    return {
        normalized
        for normalized in (_normalize_structural_text(value) for value in values)
        if normalized
    }


def _is_chunk_wrapper(normalized_label: str, metadata_values: set[str]) -> bool:
    if not normalized_label.startswith("chunk "):
        return False

    label_without_prefix = normalized_label.removeprefix("chunk ").strip()
    return label_without_prefix in metadata_values


def _normalize_structural_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(normalized.split())


def _log_chunk_graph(chunk: Chunk, generated: _GeneratedChunkGraph) -> None:
    RUN_LOGGER.info(
        "Generated chunk graph source=%s chunk=%s raw_nodes=%s kept_nodes=%s "
        "dropped_structural_nodes=%s raw_edges=%s kept_edges=%s dropped_edges=%s",
        chunk.source,
        chunk.id,
        generated.raw_nodes,
        generated.kept_nodes,
        generated.dropped_structural_nodes,
        generated.raw_edges,
        generated.kept_edges,
        generated.raw_edges - generated.kept_edges,
    )


def _log_document_graph(
    source: str,
    chunk_count: int,
    generated_chunks: Sequence[_GeneratedChunkGraph],
    graph: KnowledgeGraph,
) -> None:
    raw_nodes = sum(generated.raw_nodes for generated in generated_chunks)
    kept_nodes = sum(generated.kept_nodes for generated in generated_chunks)
    raw_edges = sum(generated.raw_edges for generated in generated_chunks)
    kept_edges = sum(generated.kept_edges for generated in generated_chunks)
    RUN_LOGGER.info(
        "Generated document graph source=%s chunks=%s raw_nodes=%s kept_nodes=%s "
        "dropped_structural_nodes=%s raw_edges=%s kept_edges=%s dropped_edges=%s "
        "final_nodes=%s final_edges=%s",
        source,
        chunk_count,
        raw_nodes,
        kept_nodes,
        sum(
            generated.dropped_structural_nodes
            for generated in generated_chunks
        ),
        raw_edges,
        kept_edges,
        raw_edges - kept_edges,
        len(graph.nodes),
        len(graph.edges),
    )


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
                    "type": _merge_node_type(existing_node, node),
                    "suggested_type": (
                        existing_node.suggested_type or node.suggested_type
                    ),
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


def _merge_node_type(existing: Node, incoming: Node) -> str:
    existing_type = normalize_type_name(existing.type)
    incoming_type = normalize_type_name(incoming.type)
    if (
        existing_type == UNCLASSIFIED_NODE_TYPE
        and incoming_type != UNCLASSIFIED_NODE_TYPE
    ):
        return incoming.type
    return existing.type
