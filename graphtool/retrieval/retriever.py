import json
import math
from collections.abc import Iterable, Sequence
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
    NodeHit,
    RelationshipHit,
    RetrievalResult,
)

BOTH_ENDPOINTS_SELECTED_BONUS = 0.25
BM25_CHUNK_WEIGHT = 1.0
SEMANTIC_CHUNK_WEIGHT = 1.0
NODE_CHUNK_SUPPORT_WEIGHT = 0.5
EDGE_CHUNK_SUPPORT_WEIGHT = 0.5
OVERLAP_CHUNK_BONUS = 0.25


def retrieve_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    top_nodes: int = 5,
    top_edges: int = 5,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
) -> RetrievalResult:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    nodes_by_id = {node.id: node for node in graph.nodes}

    node_hits = _retrieve_nodes(query, graph.nodes, chunks_by_id, top_nodes)
    relationship_hits = _retrieve_relationships(
        query,
        graph.edges,
        nodes_by_id,
        node_hits,
        top_edges,
    )
    chunk_hits = _retrieve_chunks(
        query,
        graph,
        chunks_by_id,
        nodes_by_id,
        node_hits,
        relationship_hits,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
    )
    chunk_hits = chunk_hits[:top_chunks]
    sources = _unique_ordered(hit.chunk.source for hit in chunk_hits)

    return RetrievalResult(
        query=query,
        sources=sources,
        node_hits=node_hits,
        relationship_hits=relationship_hits,
        chunks=chunk_hits,
        context_text=_format_context(query, node_hits, relationship_hits, chunk_hits),
    )


def _retrieve_nodes(
    query: str,
    nodes: Sequence[Node],
    chunks_by_id: dict[str, Chunk],
    top_nodes: int,
) -> list[NodeHit]:
    documents = [
        BM25Document(id=node.id, text=_node_text(node, chunks_by_id))
        for node in nodes
    ]
    index = BM25Index(documents)
    nodes_by_id = {node.id: node for node in nodes}

    return [
        NodeHit(
            node=nodes_by_id[document.id],
            score=score,
            matched_text=nodes_by_id[document.id].label,
        )
        for document, score in index.rank(query)
        if score > 0
    ][:top_nodes]


def _retrieve_relationships(
    query: str,
    edges: Sequence[Edge],
    nodes_by_id: dict[str, Node],
    node_hits: Sequence[NodeHit],
    top_edges: int,
) -> list[RelationshipHit]:
    selected_node_scores = {hit.node.id: hit.score for hit in node_hits}
    candidate_edges = [
        edge
        for edge in edges
        if edge.source in selected_node_scores or edge.target in selected_node_scores
    ]
    documents = [
        BM25Document(id=edge.id, text=_edge_text(edge, nodes_by_id))
        for edge in candidate_edges
    ]
    index = BM25Index(documents)
    edges_by_id = {edge.id: edge for edge in candidate_edges}

    scored: list[tuple[Edge, float]] = []
    for document, bm25_score in index.rank(query):
        edge = edges_by_id[document.id]
        source_node_score = selected_node_scores.get(edge.source, 0.0)
        target_node_score = selected_node_scores.get(edge.target, 0.0)
        both_endpoints_selected = (
            edge.source in selected_node_scores and edge.target in selected_node_scores
        )
        endpoint_bonus = (
            BOTH_ENDPOINTS_SELECTED_BONUS
            if both_endpoints_selected
            else 0.0
        )
        score = bm25_score + max(source_node_score, target_node_score) + endpoint_bonus
        if score > 0:
            scored.append((edge, score))

    scored.sort(key=lambda item: (-item[1], item[0].id))
    return [
        RelationshipHit(
            edge=edge,
            source_node=nodes_by_id[edge.source],
            target_node=nodes_by_id[edge.target],
            score=score,
            chunk_ids=list(edge.chunk_ids),
        )
        for edge, score in scored[:top_edges]
    ]


def _retrieve_chunks(
    query: str,
    graph: KnowledgeGraph,
    chunks_by_id: dict[str, Chunk],
    nodes_by_id: dict[str, Node],
    node_hits: Sequence[NodeHit],
    relationship_hits: Sequence[RelationshipHit],
    *,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
) -> list[ChunkHit]:
    linked_node_scores_by_chunk: dict[str, list[tuple[str, float]]] = {}
    linked_edge_scores_by_chunk: dict[str, list[tuple[str, float]]] = {}

    for hit in node_hits:
        for chunk_id in hit.node.chunk_ids:
            if chunk_id in chunks_by_id:
                linked_node_scores_by_chunk.setdefault(chunk_id, []).append(
                    (hit.node.id, hit.score)
                )

    for hit in relationship_hits:
        for chunk_id in hit.chunk_ids:
            if chunk_id in chunks_by_id:
                linked_edge_scores_by_chunk.setdefault(chunk_id, []).append(
                    (hit.edge.id, hit.score)
                )

    searchable_text_by_chunk = _searchable_text_by_chunk(
        graph,
        chunks_by_id,
        nodes_by_id,
    )
    documents = [
        BM25Document(id=chunk.id, text=searchable_text_by_chunk[chunk.id])
        for chunk in chunks_by_id.values()
    ]
    index = BM25Index(documents)
    bm25_scores = _normalize_scores(
        {
            document.id: score
            for document, score in index.rank(query)
        }
    )
    semantic_scores = _normalize_scores(
        _semantic_chunk_scores(
            query,
            searchable_text_by_chunk,
            embedding_client,
            chunk_embedding_store,
        )
    )

    chunk_hits: list[ChunkHit] = []
    for chunk in chunks_by_id.values():
        node_scores = linked_node_scores_by_chunk.get(chunk.id, [])
        edge_scores = linked_edge_scores_by_chunk.get(chunk.id, [])
        best_node_score = max((score for _, score in node_scores), default=0.0)
        best_edge_score = max((score for _, score in edge_scores), default=0.0)
        overlap_bonus = OVERLAP_CHUNK_BONUS if node_scores and edge_scores else 0.0
        score = (
            bm25_scores.get(chunk.id, 0.0) * BM25_CHUNK_WEIGHT
            + semantic_scores.get(chunk.id, 0.0) * SEMANTIC_CHUNK_WEIGHT
            + best_node_score * NODE_CHUNK_SUPPORT_WEIGHT
            + best_edge_score * EDGE_CHUNK_SUPPORT_WEIGHT
            + overlap_bonus
        )
        if score <= 0:
            continue
        chunk_hits.append(
            ChunkHit(
                chunk=chunk,
                score=score,
                linked_node_ids=_unique_ordered(node_id for node_id, _ in node_scores),
                linked_edge_ids=_unique_ordered(edge_id for edge_id, _ in edge_scores),
            )
        )

    chunk_hits.sort(key=lambda hit: (-hit.score, hit.chunk.index, hit.chunk.id))
    return chunk_hits


def _searchable_text_by_chunk(
    graph: KnowledgeGraph,
    chunks_by_id: dict[str, Chunk],
    nodes_by_id: dict[str, Node],
) -> dict[str, str]:
    nodes_by_chunk: dict[str, list[Node]] = {}
    for node in graph.nodes:
        for chunk_id in node.chunk_ids:
            if chunk_id in chunks_by_id:
                nodes_by_chunk.setdefault(chunk_id, []).append(node)

    edges_by_chunk: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
            continue
        for chunk_id in edge.chunk_ids:
            if chunk_id in chunks_by_id:
                edges_by_chunk.setdefault(chunk_id, []).append(edge)

    return {
        chunk.id: _chunk_text(
            chunk,
            nodes_by_chunk.get(chunk.id, []),
            edges_by_chunk.get(chunk.id, []),
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


def _node_text(node: Node, chunks_by_id: dict[str, Chunk]) -> str:
    headings = [
        " ".join(chunks_by_id[chunk_id].heading_path)
        for chunk_id in node.chunk_ids
        if chunk_id in chunks_by_id
    ]
    return " ".join(
        [
            node.id,
            node.label,
            node.type,
            node.suggested_type or "",
            _properties_text(node.properties),
            *headings,
        ]
    )


def _edge_text(edge: Edge, nodes_by_id: dict[str, Node]) -> str:
    source = nodes_by_id[edge.source]
    target = nodes_by_id[edge.target]
    return " ".join(
        [
            source.label,
            source.type,
            edge.label,
            _properties_text(edge.properties),
            target.label,
            target.type,
        ]
    )


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
        lines.extend(
            f"{node.label} [{node.type}]"
            for node in nodes
        )

    if edges:
        lines.append("Relationships:")
        lines.extend(
            _relationship_text(edge, nodes_by_id)
            for edge in edges
        )

    return "\n".join(lines)


def _relationship_text(edge: Edge, nodes_by_id: dict[str, Node]) -> str:
    source = nodes_by_id[edge.source]
    target = nodes_by_id[edge.target]
    return f"{source.label} --{edge.label}--> {target.label}"


def _properties_text(properties: dict[str, Any]) -> str:
    if not properties:
        return ""
    return json.dumps(properties, sort_keys=True)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _format_context(
    query: str,
    node_hits: Sequence[NodeHit],
    relationship_hits: Sequence[RelationshipHit],
    chunk_hits: Sequence[ChunkHit],
) -> str:
    lines = [f"Query: {query}", "", "Relevant nodes:"]
    if node_hits:
        lines.extend(
            f"- {hit.node.label} [{hit.node.type}] ({hit.node.id})"
            for hit in node_hits
        )
    else:
        lines.append("- None")

    lines.extend(["", "Relevant relationships:"])
    if relationship_hits:
        lines.extend(
            "- "
            f"{hit.source_node.label} --{hit.edge.label}--> {hit.target_node.label} "
            f"({hit.edge.id})"
            for hit in relationship_hits
        )
    else:
        lines.append("- None")

    lines.extend(["", "Evidence:"])
    if chunk_hits:
        for hit in chunk_hits:
            heading = " > ".join(hit.chunk.heading_path)
            metadata = f"{hit.chunk.id} | {hit.chunk.source}"
            if heading:
                metadata = f"{metadata} | {heading}"
            lines.extend([f"[{metadata}]", hit.chunk.text])
    else:
        lines.append("- None")

    return "\n".join(lines)


def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
