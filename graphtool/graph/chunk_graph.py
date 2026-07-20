import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from graphtool.chunking.types import Chunk
from graphtool.graph.extraction_store import (
    ExtractedEdge,
    ExtractedKnowledgeGraph,
    ExtractedNode,
)
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)
_DROPPED_EDGES_LOCK = Lock()

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
class GeneratedChunkGraph:
    graph: KnowledgeGraph
    raw_nodes: int
    kept_nodes: int
    dropped_structural_nodes: int
    raw_edges: int
    kept_edges: int


def build_chunk_graph(
    chunk: Chunk,
    graph: ExtractedKnowledgeGraph,
    *,
    dropped_edges_path: Path | None = None,
) -> GeneratedChunkGraph:
    structural_node_refs = _structural_node_refs(graph.nodes, chunk)
    nodes, node_id_by_ref = _build_chunk_nodes(
        graph.nodes,
        structural_node_refs,
        chunk,
    )
    edges_by_key: dict[tuple[str, str, str], Edge] = {}
    for edge in graph.edges:
        if (
            edge.source_ref in structural_node_refs
            or edge.target_ref in structural_node_refs
        ):
            continue

        missing = []
        if edge.source_ref not in node_id_by_ref:
            missing.append("source")
        if edge.target_ref not in node_id_by_ref:
            missing.append("target")
        if missing:
            _record_dropped_edge(chunk, edge, missing, dropped_edges_path)
            continue

        source = node_id_by_ref[edge.source_ref]
        target = node_id_by_ref[edge.target_ref]
        key = (source, target, edge.label)
        edges_by_key.setdefault(
            key,
            Edge(
                id=edge.id,
                source=source,
                target=target,
                label=edge.label,
                chunk_ids=[chunk.id],
            ),
        )

    edges = [
        edge.model_copy(update={"id": f"edge-{index:04d}"})
        for index, edge in enumerate(edges_by_key.values(), start=1)
    ]
    generated = GeneratedChunkGraph(
        graph=KnowledgeGraph(nodes=nodes, edges=edges),
        raw_nodes=len(graph.nodes),
        kept_nodes=len(nodes),
        dropped_structural_nodes=len(structural_node_refs),
        raw_edges=len(graph.edges),
        kept_edges=len(edges),
    )
    _log_chunk_graph(chunk, generated)
    return generated


def log_document_graph(
    source: str,
    chunk_count: int,
    generated_chunks: Sequence[GeneratedChunkGraph],
    graph: KnowledgeGraph,
    *,
    cached_chunks: int,
    generated_chunk_count: int,
    extraction_requests: int,
) -> None:
    raw_nodes = sum(generated.raw_nodes for generated in generated_chunks)
    kept_nodes = sum(generated.kept_nodes for generated in generated_chunks)
    raw_edges = sum(generated.raw_edges for generated in generated_chunks)
    kept_edges = sum(generated.kept_edges for generated in generated_chunks)
    RUN_LOGGER.debug(
        "Generated document graph source=%s chunks=%s raw_nodes=%s kept_nodes=%s "
        "dropped_structural_nodes=%s raw_edges=%s kept_edges=%s dropped_edges=%s "
        "final_nodes=%s final_edges=%s cached_chunks=%s generated_chunks=%s "
        "extraction_requests=%s",
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
        cached_chunks,
        generated_chunk_count,
        extraction_requests,
    )


def _build_chunk_nodes(
    extracted_nodes: Sequence[ExtractedNode],
    structural_node_refs: set[str],
    chunk: Chunk,
) -> tuple[list[Node], dict[str, str]]:
    nodes: list[Node] = []
    node_id_by_ref: dict[str, str] = {}
    for extracted_node in extracted_nodes:
        if extracted_node.ref in structural_node_refs:
            continue

        node_id = f"{chunk.id}::node-{len(nodes) + 1:04d}"
        node_id_by_ref[extracted_node.ref] = node_id
        nodes.append(
            Node(
                id=node_id,
                label=extracted_node.label,
                type=extracted_node.type,
                suggested_type=extracted_node.suggested_type,
                chunk_ids=[chunk.id],
            )
        )
    return nodes, node_id_by_ref


def _structural_node_refs(
    nodes: Sequence[ExtractedNode],
    chunk: Chunk,
) -> set[str]:
    return {node.ref for node in nodes if _is_structural_node(node, chunk)}


def _is_structural_node(node: ExtractedNode, chunk: Chunk) -> bool:
    normalized_type = _normalize_structural_text(node.type)
    if normalized_type in _STRUCTURAL_NODE_TYPES:
        return True

    normalized_label = _normalize_structural_text(node.label)
    if normalized_label in _STRUCTURAL_NODE_LABELS:
        return True
    if normalized_label in _source_metadata_values(chunk):
        return True
    return _is_chunk_wrapper(normalized_label, _heading_metadata_values(chunk))


def _source_metadata_values(chunk: Chunk) -> set[str]:
    return _normalized_values({chunk.id, chunk.source})


def _heading_metadata_values(chunk: Chunk) -> set[str]:
    heading_path = " > ".join(chunk.heading_path) if chunk.heading_path else "(none)"
    return _normalized_values({heading_path, *chunk.heading_path})


def _normalized_values(values: set[str]) -> set[str]:
    return {
        normalized
        for normalized in (_normalize_structural_text(value) for value in values)
        if normalized
    }


def _is_chunk_wrapper(normalized_label: str, metadata_values: set[str]) -> bool:
    if not normalized_label.startswith("chunk "):
        return False
    return normalized_label.removeprefix("chunk ").strip() in metadata_values


def _normalize_structural_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(normalized.split())


def _log_chunk_graph(chunk: Chunk, generated: GeneratedChunkGraph) -> None:
    RUN_LOGGER.debug(
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


def _record_dropped_edge(
    chunk: Chunk,
    edge: ExtractedEdge,
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

    with _DROPPED_EDGES_LOCK:
        dropped_edges_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": chunk.source,
            "chunk_id": chunk.id,
            "edge_id": edge.id,
            "label": edge.label,
            "edge_source": edge.source_ref,
            "edge_target": edge.target_ref,
            "missing": missing,
        }
        with dropped_edges_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def _missing_edge_description(edge: ExtractedEdge, missing: list[str]) -> str:
    parts = []
    if "source" in missing:
        parts.append(f"source node {edge.source_ref}")
    if "target" in missing:
        parts.append(f"target node {edge.target_ref}")
    return ", ".join(parts)
