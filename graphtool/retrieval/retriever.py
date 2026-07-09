import json
from collections.abc import Iterable, Sequence
from typing import Any

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.retrieval.bm25 import BM25Document, BM25Index
from graphtool.retrieval.types import (
    ChunkHit,
    NodeHit,
    RelationshipHit,
    RetrievalResult,
)

BOTH_ENDPOINTS_SELECTED_BONUS = 0.25
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
    chunk_hits = _retrieve_chunks(query, chunks_by_id, node_hits, relationship_hits)
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
    chunks_by_id: dict[str, Chunk],
    node_hits: Sequence[NodeHit],
    relationship_hits: Sequence[RelationshipHit],
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

    candidate_chunk_ids = set(linked_node_scores_by_chunk) | set(
        linked_edge_scores_by_chunk
    )
    candidate_chunks = [chunks_by_id[chunk_id] for chunk_id in candidate_chunk_ids]
    documents = [
        BM25Document(id=chunk.id, text=_chunk_text(chunk))
        for chunk in candidate_chunks
    ]
    index = BM25Index(documents)

    chunk_hits: list[ChunkHit] = []
    for document, direct_score in index.rank(query):
        node_scores = linked_node_scores_by_chunk.get(document.id, [])
        edge_scores = linked_edge_scores_by_chunk.get(document.id, [])
        best_node_score = max((score for _, score in node_scores), default=0.0)
        best_edge_score = max((score for _, score in edge_scores), default=0.0)
        overlap_bonus = OVERLAP_CHUNK_BONUS if node_scores and edge_scores else 0.0
        score = (
            direct_score
            + best_node_score * NODE_CHUNK_SUPPORT_WEIGHT
            + best_edge_score * EDGE_CHUNK_SUPPORT_WEIGHT
            + overlap_bonus
        )
        if score <= 0:
            continue
        chunk_hits.append(
            ChunkHit(
                chunk=chunks_by_id[document.id],
                score=score,
                linked_node_ids=_unique_ordered(node_id for node_id, _ in node_scores),
                linked_edge_ids=_unique_ordered(edge_id for edge_id, _ in edge_scores),
            )
        )

    chunk_hits.sort(key=lambda hit: (-hit.score, hit.chunk.index, hit.chunk.id))
    return chunk_hits


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


def _chunk_text(chunk: Chunk) -> str:
    return " ".join([*chunk.heading_path, chunk.text])


def _properties_text(properties: dict[str, Any]) -> str:
    if not properties:
        return ""
    return json.dumps(properties, sort_keys=True)


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
