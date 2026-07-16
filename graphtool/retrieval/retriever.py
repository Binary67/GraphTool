import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.bm25 import BM25Document, BM25Index
from graphtool.retrieval.embedding_store import (
    ChunkEmbeddingRecord,
    ChunkEmbeddingStore,
    chunk_embedding_input_hash,
)
from graphtool.retrieval.types import (
    ChunkHit,
    ChunkRelationship,
    GraphPathHit,
    RetrievalResult,
    SourceReference,
)

PRIMARY_LABEL_BM25_WEIGHT = 2.0
ALIAS_BM25_WEIGHT = 1.5
CONTENT_BM25_WEIGHT = 1.0
METADATA_BM25_WEIGHT = 0.5
SEMANTIC_CHUNK_WEIGHT = 1.0


@dataclass(frozen=True)
class _ChunkSearchFields:
    primary_labels: str
    aliases: str
    content: str
    metadata: str


@dataclass(frozen=True)
class _ChunkGraphIndex:
    nodes_by_chunk: dict[str, list[Node]]
    edges_by_chunk: dict[str, list[Edge]]


def _build_chunk_graph_index(
    graph: KnowledgeGraph,
    chunks_by_id: dict[str, Chunk],
    nodes_by_id: dict[str, Node],
) -> _ChunkGraphIndex:
    nodes_by_chunk: dict[str, list[Node]] = {}
    for node in sorted(graph.nodes, key=lambda item: item.id):
        for chunk_id in node.chunk_ids:
            if chunk_id in chunks_by_id:
                nodes_by_chunk.setdefault(chunk_id, []).append(node)

    edges_by_chunk: dict[str, list[Edge]] = {}
    for edge in sorted(graph.edges, key=lambda item: item.id):
        if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
            continue
        for chunk_id in edge.chunk_ids:
            if chunk_id in chunks_by_id:
                edges_by_chunk.setdefault(chunk_id, []).append(edge)

    return _ChunkGraphIndex(
        nodes_by_chunk=nodes_by_chunk,
        edges_by_chunk=edges_by_chunk,
    )


def retrieve_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
) -> RetrievalResult:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    nodes_by_id = {node.id: node for node in graph.nodes}
    index = _build_chunk_graph_index(graph, chunks_by_id, nodes_by_id)
    ranked_chunks = _rank_chunks(
        query,
        chunks_by_id,
        index,
        nodes_by_id,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
    )[:top_chunks]
    chunk_hits = _attach_graph_annotations(ranked_chunks, index, nodes_by_id)
    sources = _unique_ordered(hit.chunk.source for hit in chunk_hits)

    return RetrievalResult(
        query=query,
        sources=sources,
        references=_source_references(chunk_hits),
        chunks=chunk_hits,
        context_text=_format_context(query, chunk_hits),
    )


def _rank_chunks(
    query: str,
    chunks_by_id: dict[str, Chunk],
    index: _ChunkGraphIndex,
    nodes_by_id: dict[str, Node],
    *,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
) -> list[tuple[Chunk, float]]:
    search_fields_by_chunk = _search_fields_by_chunk(chunks_by_id, index)
    searchable_text_by_chunk = _searchable_text_by_chunk(
        chunks_by_id, index, nodes_by_id
    )
    primary_label_scores = _bm25_scores(
        query,
        {
            chunk_id: fields.primary_labels
            for chunk_id, fields in search_fields_by_chunk.items()
        },
    )
    alias_scores = _bm25_scores(
        query,
        {
            chunk_id: fields.aliases
            for chunk_id, fields in search_fields_by_chunk.items()
        },
    )
    content_scores = _bm25_scores(
        query,
        {
            chunk_id: fields.content
            for chunk_id, fields in search_fields_by_chunk.items()
        },
    )
    metadata_scores = _bm25_scores(
        query,
        {
            chunk_id: fields.metadata
            for chunk_id, fields in search_fields_by_chunk.items()
        },
    )
    semantic_scores = _normalize_scores(
        _semantic_chunk_scores(
            query,
            searchable_text_by_chunk,
            embedding_client,
            chunk_embedding_store,
        )
    )

    ranked = []
    for chunk in chunks_by_id.values():
        score = (
            primary_label_scores.get(chunk.id, 0.0) * PRIMARY_LABEL_BM25_WEIGHT
            + alias_scores.get(chunk.id, 0.0) * ALIAS_BM25_WEIGHT
            + content_scores.get(chunk.id, 0.0) * CONTENT_BM25_WEIGHT
            + metadata_scores.get(chunk.id, 0.0) * METADATA_BM25_WEIGHT
            + semantic_scores.get(chunk.id, 0.0) * SEMANTIC_CHUNK_WEIGHT
        )
        if score > 0:
            ranked.append((chunk, score))

    ranked.sort(key=lambda item: (-item[1], item[0].index, item[0].id))
    return ranked


def _bm25_scores(query: str, text_by_chunk: dict[str, str]) -> dict[str, float]:
    index = BM25Index(
        [
            BM25Document(id=chunk_id, text=text)
            for chunk_id, text in text_by_chunk.items()
        ]
    )
    return _normalize_scores(
        {
            document.id: score
            for document, score in index.rank(query)
        }
    )


def _search_fields_by_chunk(
    chunks_by_id: dict[str, Chunk],
    index: _ChunkGraphIndex,
) -> dict[str, _ChunkSearchFields]:
    fields_by_chunk = {}
    for chunk in chunks_by_id.values():
        nodes = index.nodes_by_chunk.get(chunk.id, [])
        edges = index.edges_by_chunk.get(chunk.id, [])
        fields_by_chunk[chunk.id] = _ChunkSearchFields(
            primary_labels="\n".join(
                _unique_search_text(node.label for node in nodes)
            ),
            aliases="\n".join(
                _unique_search_text(
                    alias
                    for node in nodes
                    for alias in node.aliases
                )
            ),
            content=chunk.text,
            metadata="\n".join(
                _unique_search_text(
                    [
                        *chunk.heading_path,
                        *(node.type for node in nodes),
                        *(
                            node.suggested_type
                            for node in nodes
                            if node.suggested_type is not None
                        ),
                        *(
                            _properties_text(node.properties)
                            for node in nodes
                            if node.properties
                        ),
                        *(
                            _relationship_metadata_text(edge)
                            for edge in edges
                        ),
                    ]
                )
            ),
        )
    return fields_by_chunk


def _relationship_metadata_text(edge: Edge) -> str:
    if not edge.properties:
        return edge.label
    return f"{edge.label} | properties: {_properties_text(edge.properties)}"


def _unique_search_text(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = " ".join(value.casefold().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def _attach_graph_annotations(
    ranked_chunks: Sequence[tuple[Chunk, float]],
    index: _ChunkGraphIndex,
    nodes_by_id: dict[str, Node],
) -> list[ChunkHit]:
    relationships_by_chunk: dict[str, list[ChunkRelationship]] = {}
    for chunk, _ in ranked_chunks:
        for edge in index.edges_by_chunk.get(chunk.id, []):
            relationship = ChunkRelationship(
                edge=edge,
                source_node=nodes_by_id[edge.source],
                target_node=nodes_by_id[edge.target],
            )
            relationships_by_chunk.setdefault(chunk.id, []).append(relationship)

    return [
        ChunkHit(
            chunk=chunk,
            score=score,
            linked_nodes=index.nodes_by_chunk.get(chunk.id, []),
            linked_relationships=relationships_by_chunk.get(chunk.id, []),
        )
        for chunk, score in ranked_chunks
    ]


def _searchable_text_by_chunk(
    chunks_by_id: dict[str, Chunk],
    index: _ChunkGraphIndex,
    nodes_by_id: dict[str, Node],
) -> dict[str, str]:
    return {
        chunk.id: _chunk_text(
            chunk,
            index.nodes_by_chunk.get(chunk.id, []),
            index.edges_by_chunk.get(chunk.id, []),
            nodes_by_id,
        )
        for chunk in chunks_by_id.values()
    }


def _semantic_chunk_scores(
    query: str,
    searchable_text_by_chunk: dict[str, str],
    embedding_client: EmbeddingClient | None,
    chunk_embedding_store: ChunkEmbeddingStore | None,
) -> dict[str, float]:
    if embedding_client is None or not searchable_text_by_chunk:
        return {}

    query_vector = embedding_client.embed_texts([query])[0]
    records = chunk_embedding_store.load() if chunk_embedding_store is not None else {}
    embedding_model = embedding_client.embedding_model
    records_to_save = dict(records)
    chunk_records: dict[str, ChunkEmbeddingRecord] = {}
    missing: list[tuple[str, str, str]] = []

    for chunk_id, text in searchable_text_by_chunk.items():
        text_hash = chunk_embedding_input_hash(text)
        record = records.get(chunk_id)
        if (
            record is not None
            and record.embedding_model == embedding_model
            and record.embedding_input_hash == text_hash
        ):
            chunk_records[chunk_id] = record
            continue

        missing.append((chunk_id, text_hash, text))

    if missing:
        vectors = embedding_client.embed_texts([text for _, _, text in missing])
        for (chunk_id, text_hash, _), vector in zip(missing, vectors, strict=True):
            record = ChunkEmbeddingRecord(
                chunk_id=chunk_id,
                embedding_model=embedding_model,
                embedding_input_hash=text_hash,
                vector=vector,
            )
            records_to_save[chunk_id] = record
            chunk_records[chunk_id] = record
        if chunk_embedding_store is not None:
            chunk_embedding_store.save(records_to_save)

    return {
        chunk_id: _cosine_similarity(query_vector, record.vector)
        for chunk_id, record in chunk_records.items()
    }


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    positive_scores = {
        key: score
        for key, score in scores.items()
        if score > 0
    }
    if not positive_scores:
        return {}

    max_score = max(positive_scores.values())
    return {
        key: score / max_score
        for key, score in positive_scores.items()
    }


def _chunk_text(
    chunk: Chunk,
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    nodes_by_id: dict[str, Node],
) -> str:
    lines = []
    if chunk.heading_path:
        lines.extend(["Heading:", " > ".join(chunk.heading_path)])

    lines.extend(["Content:", chunk.text])

    if nodes:
        lines.append("Entities:")
        lines.extend(_node_text(node) for node in nodes)

    if edges:
        lines.append("Relationships:")
        lines.extend(
            _relationship_text(
                edge,
                nodes_by_id[edge.source],
                nodes_by_id[edge.target],
            )
            for edge in edges
        )

    return "\n".join(lines)


def _node_text(node: Node) -> str:
    parts = [f"{node.label} [{node.type}]"]
    if node.aliases:
        parts.append(f"aliases: {', '.join(node.aliases)}")
    if node.suggested_type:
        parts.append(f"suggested type: {node.suggested_type}")
    if node.properties:
        parts.append(f"properties: {_properties_text(node.properties)}")
    return " | ".join(parts)


def _relationship_text(edge: Edge, source: Node, target: Node) -> str:
    text = f"{source.label} --{edge.label}--> {target.label}"
    if edge.properties:
        return f"{text} | properties: {_properties_text(edge.properties)}"
    return text


def _properties_text(properties: dict[str, Any]) -> str:
    return json.dumps(properties, sort_keys=True)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return sum(
        a * b
        for a, b in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


def _format_context(
    query: str,
    chunk_hits: Sequence[ChunkHit],
    graph_paths: Sequence[GraphPathHit] = (),
) -> str:
    lines = [f"Query: {query}", "", "Evidence:"]
    if not chunk_hits:
        lines.append("- None")
    else:
        for hit in chunk_hits:
            heading = " > ".join(hit.chunk.heading_path)
            metadata = f"{hit.chunk.id} | {hit.chunk.source}"
            page_reference = _format_page_range(
                hit.chunk.page_start,
                hit.chunk.page_end,
            )
            if page_reference:
                metadata = f"{metadata} | {page_reference}"
            if heading:
                metadata = f"{metadata} | {heading}"
            lines.extend([f"[{metadata}]", hit.chunk.text])

            if hit.linked_nodes:
                lines.append("Linked entities:")
                lines.extend(f"- {_node_text(node)}" for node in hit.linked_nodes)

            if hit.linked_relationships:
                lines.append("Linked relationships:")
                lines.extend(
                    "- " + _relationship_text(
                        relationship.edge,
                        relationship.source_node,
                        relationship.target_node,
                    )
                    for relationship in hit.linked_relationships
                )

            lines.append("")

    if graph_paths:
        lines.extend(["", "Graph paths:"])
        for path in graph_paths:
            lines.append(f"- {_graph_path_text(path)}")
            if path.chunk_ids:
                lines.append(f"  Evidence chunks: {', '.join(path.chunk_ids)}")

    return "\n".join(lines).rstrip()


def _graph_path_text(path: GraphPathHit) -> str:
    parts = [path.nodes[0].label]
    for index, edge in enumerate(path.edges):
        left = path.nodes[index]
        right = path.nodes[index + 1]
        if edge.source == left.id and edge.target == right.id:
            parts.append(f"--{edge.label}--> {right.label}")
        else:
            parts.append(f"<--{edge.label}-- {right.label}")
    return " ".join(parts)


def _source_references(chunk_hits: Sequence[ChunkHit]) -> list[SourceReference]:
    ranges_by_source: dict[str, list[tuple[int, int]]] = {}
    unpaged_sources = set()
    for hit in chunk_hits:
        source = hit.chunk.source
        ranges_by_source.setdefault(source, [])
        if hit.chunk.page_start is None:
            unpaged_sources.add(source)
        else:
            assert hit.chunk.page_end is not None
            ranges_by_source[source].append(
                (hit.chunk.page_start, hit.chunk.page_end)
            )

    references = []
    for source, ranges in ranges_by_source.items():
        if source in unpaged_sources:
            references.append(SourceReference(source=source))
            continue

        merged = []
        for page_start, page_end in sorted(ranges):
            if merged and page_start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], page_end))
            else:
                merged.append((page_start, page_end))
        references.extend(
            SourceReference(
                source=source,
                page_start=page_start,
                page_end=page_end,
            )
            for page_start, page_end in merged
        )
    return references


def _format_page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return ""
    if page_start == page_end:
        return f"p. {page_start}"
    return f"pp. {page_start}-{page_end}"


def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
