import hashlib
import json
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Protocol

from pydantic import ValidationError

from graphtool.chunking.types import Chunk
from graphtool.graph.extraction_store import (
    ChunkExtractionStore,
    ExtractedEdge as _ExtractedEdge,
    ExtractedKnowledgeGraph as _ExtractedKnowledgeGraph,
    ExtractedNode as _ExtractedNode,
)
from graphtool.graph.provenance import (
    add_edge_provenance,
    add_node_provenance,
    merge_edges,
    merge_nodes,
)
from graphtool.graph.taxonomy import (
    TaxonomySuggestionStore,
    UNCLASSIFIED_NODE_TYPE,
    canonical_node_type_text,
    make_taxonomy_suggestion_records,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)
_DROPPED_EDGES_LOCK = Lock()


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
    "mechanics. Prefer a small graph of the most salient entities. Assign every "
    "node a unique temporary ref within this response. Every edge must use "
    "source_ref and target_ref to reference existing node refs. Node type must "
    "be one of: "
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


@dataclass(frozen=True)
class _ChunkExtractions:
    graphs: list[_ExtractedKnowledgeGraph]
    records: dict[str, _ExtractedKnowledgeGraph]
    cached_chunks: int
    generated_chunks: int
    extraction_requests: int


def generate_knowledge_graph(
    chunks: Sequence[Chunk],
    source: str,
    llm: LLMClient,
    *,
    content_hash: str,
    resolver: GraphResolver | None = None,
    dropped_edges_path: Path | None = None,
    taxonomy_suggestion_store: TaxonomySuggestionStore | None = None,
    extraction_store: ChunkExtractionStore | None = None,
    max_workers: int = 4,
) -> KnowledgeGraph:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    extractions = _extract_chunks(
        chunks,
        source,
        llm,
        extraction_store,
        max_workers=max_workers,
    )
    generated_chunks = [
        _build_chunk_graph(
            chunk,
            extracted,
            dropped_edges_path=dropped_edges_path,
        )
        for chunk, extracted in zip(chunks, extractions.graphs, strict=True)
    ]

    taxonomy_suggestion_records = []
    for chunk, generated in zip(chunks, generated_chunks, strict=True):
        if taxonomy_suggestion_store is not None:
            taxonomy_suggestion_records.extend(
                make_taxonomy_suggestion_records(
                    nodes=generated.graph.nodes,
                    source=chunk.source,
                    chunk_id=chunk.id,
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
    if extraction_store is not None:
        extraction_store.replace(source, extractions.records)
    _log_document_graph(
        source,
        len(chunks),
        generated_chunks,
        graph,
        cached_chunks=extractions.cached_chunks,
        generated_chunk_count=extractions.generated_chunks,
        extraction_requests=extractions.extraction_requests,
    )
    return graph.model_copy(
        update={
            "metadata": GraphMetadata(
                source=source,
                content_hash=content_hash,
                model=None,
                created_at=datetime.now(timezone.utc),
            )
        }
    )


def _extract_chunks(
    chunks: Sequence[Chunk],
    source: str,
    llm: LLMClient,
    extraction_store: ChunkExtractionStore | None,
    *,
    max_workers: int,
) -> _ChunkExtractions:
    messages_by_chunk = [_chunk_messages(chunk) for chunk in chunks]
    extract = partial(_extract_chunk, llm=llm)

    if extraction_store is None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            graphs = list(executor.map(extract, chunks, messages_by_chunk))
        return _ChunkExtractions(
            graphs=graphs,
            records={},
            cached_chunks=0,
            generated_chunks=len(chunks),
            extraction_requests=len(chunks),
        )

    cached_records = extraction_store.load(source)
    cache_keys = [
        _extraction_cache_key(messages, llm.text_model)
        for messages in messages_by_chunk
    ]
    records = {
        cache_key: cached_records[cache_key]
        for cache_key in dict.fromkeys(cache_keys)
        if cache_key in cached_records
    }
    missing = {}
    for cache_key, chunk, messages in zip(
        cache_keys,
        chunks,
        messages_by_chunk,
        strict=True,
    ):
        if cache_key not in records and cache_key not in missing:
            missing[cache_key] = (chunk, messages)

    missing_items = list(missing.items())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        generated = list(
            executor.map(
                extract,
                (item[1][0] for item in missing_items),
                (item[1][1] for item in missing_items),
            )
        )
    records.update(
        (item[0], graph)
        for item, graph in zip(missing_items, generated, strict=True)
    )
    cached_chunks = sum(cache_key in cached_records for cache_key in cache_keys)
    return _ChunkExtractions(
        graphs=[records[cache_key] for cache_key in cache_keys],
        records={
            cache_key: records[cache_key]
            for cache_key in dict.fromkeys(cache_keys)
        },
        cached_chunks=cached_chunks,
        generated_chunks=len(chunks) - cached_chunks,
        extraction_requests=len(missing_items),
    )


def _chunk_messages(chunk: Chunk) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=USER_PROMPT_TEMPLATE.format(
                heading_path=_heading_path_text(chunk),
                markdown=chunk.text,
            ),
        ),
    ]


def _extraction_cache_key(
    messages: Sequence[LLMMessage],
    text_model: str,
) -> str:
    payload = {
        "messages": [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        "response_schema": _ExtractedKnowledgeGraph.model_json_schema(),
        "text_model": text_model,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _extract_chunk(
    chunk: Chunk,
    messages: Sequence[LLMMessage],
    *,
    llm: LLMClient,
) -> _ExtractedKnowledgeGraph:
    try:
        graph = llm.generate_structured(messages, _ExtractedKnowledgeGraph)
    except ValidationError:
        RUN_LOGGER.warning(
            "Retrying chunk graph generation after invalid structured response: %s",
            chunk.id,
        )
        graph = llm.generate_structured(messages, _ExtractedKnowledgeGraph)
    return graph


def _build_chunk_graph(
    chunk: Chunk,
    graph: _ExtractedKnowledgeGraph,
    *,
    dropped_edges_path: Path | None = None,
) -> _GeneratedChunkGraph:
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
    generated = _GeneratedChunkGraph(
        graph=KnowledgeGraph(nodes=nodes, edges=edges),
        raw_nodes=len(graph.nodes),
        kept_nodes=len(nodes),
        dropped_structural_nodes=len(structural_node_refs),
        raw_edges=len(graph.edges),
        kept_edges=len(edges),
    )
    _log_chunk_graph(chunk, generated)
    return generated


def _build_chunk_nodes(
    extracted_nodes: Sequence[_ExtractedNode],
    structural_node_refs: set[str],
    chunk: Chunk,
) -> tuple[list[Node], dict[str, str]]:
    nodes: list[Node] = []
    node_id_by_ref: dict[str, str] = {}
    for extracted_node in extracted_nodes:
        if extracted_node.ref in structural_node_refs:
            continue

        node_id = _scoped_node_id(chunk.id, len(nodes) + 1)
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


def _scoped_node_id(chunk_id: str, index: int) -> str:
    return f"{chunk_id}::node-{index:04d}"


def _heading_path_text(chunk: Chunk) -> str:
    if not chunk.heading_path:
        return "(none)"
    return " > ".join(chunk.heading_path)


def _structural_node_refs(nodes: Sequence[_ExtractedNode], chunk: Chunk) -> set[str]:
    return {
        node.ref
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
    if normalized_label in metadata_values:
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
    *,
    cached_chunks: int,
    generated_chunk_count: int,
    extraction_requests: int,
) -> None:
    raw_nodes = sum(generated.raw_nodes for generated in generated_chunks)
    kept_nodes = sum(generated.kept_nodes for generated in generated_chunks)
    raw_edges = sum(generated.raw_edges for generated in generated_chunks)
    kept_edges = sum(generated.kept_edges for generated in generated_chunks)
    RUN_LOGGER.info(
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


def _missing_edge_description(edge: _ExtractedEdge, missing: list[str]) -> str:
    parts = []
    if "source" in missing:
        parts.append(f"source node {edge.source_ref}")
    if "target" in missing:
        parts.append(f"target node {edge.target_ref}")
    return ", ".join(parts)


def combine_knowledge_graphs(graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
    nodes_by_id: dict[str, Node] = {}
    edges_by_key: dict[tuple[str, str, str], Edge] = {}
    used_edge_ids: set[str] = set()
    next_edge_index = 1

    for graph in graphs:
        for node in graph.nodes:
            node = add_node_provenance(node, graph.metadata)
            existing_node = nodes_by_id.get(node.id)
            if existing_node is None:
                nodes_by_id[node.id] = node
                continue

            nodes_by_id[node.id] = merge_nodes(existing_node, node)

        for edge in graph.edges:
            edge = add_edge_provenance(edge, graph.metadata)
            key = (edge.source, edge.target, edge.label)
            existing_edge = edges_by_key.get(key)
            if existing_edge is None:
                if not edge.provenance or edge.id in used_edge_ids:
                    edge_id, next_edge_index = _next_edge_id(
                        used_edge_ids,
                        next_edge_index,
                    )
                    edge = edge.model_copy(update={"id": edge_id})
                else:
                    used_edge_ids.add(edge.id)
                edges_by_key[key] = edge
                continue

            edges_by_key[key] = merge_edges(existing_edge, edge)

    return KnowledgeGraph(
        nodes=list(nodes_by_id.values()),
        edges=list(edges_by_key.values()),
    )


def _next_edge_id(used_ids: set[str], start_index: int) -> tuple[str, int]:
    index = start_index
    while True:
        edge_id = f"edge-{index:04d}"
        index += 1
        if edge_id not in used_ids:
            used_ids.add(edge_id)
            return edge_id, index
