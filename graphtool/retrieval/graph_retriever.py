import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from graphtool.chunking.types import Chunk
from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.bm25 import BM25Document, BM25Index, tokenize
from graphtool.retrieval.retriever import (
    _attach_graph_annotations,
    _build_chunk_graph_index,
    _cosine_similarity,
    _format_context,
    _source_references,
    _unique_ordered,
)
from graphtool.retrieval.types import GraphPathHit, RetrievalResult

DEFAULT_MAX_HOPS = 2
DEFAULT_TOP_PATHS = 5
DEFAULT_TOP_SEEDS = 5
DEFAULT_BEAM_WIDTH = 50
PATH_HOP_DECAY = 0.85


class NodeEmbeddingStore(Protocol):
    def load(self) -> dict[str, NodeEmbeddingRecord]:
        ...


@dataclass(frozen=True)
class _PathCandidate:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    seed_score: float


def retrieve_graph_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    node_embedding_store: NodeEmbeddingStore | None = None,
) -> RetrievalResult:
    if max_hops < 1:
        raise ValueError("max_hops must be positive")
    if top_paths < 1:
        raise ValueError("top_paths must be positive")
    if top_chunks < 1:
        raise ValueError("top_chunks must be positive")

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    nodes_by_id = {node.id: node for node in graph.nodes}
    edges_by_id = {edge.id: edge for edge in graph.edges}
    mention_scores = _mentioned_node_scores(query, graph.nodes)
    seed_scores = _seed_node_scores(
        query,
        graph.nodes,
        mention_scores,
        embedding_client,
        node_embedding_store,
    )
    candidates = _traverse_paths(
        query,
        graph,
        chunks_by_id,
        nodes_by_id,
        edges_by_id,
        seed_scores,
        mention_scores,
        max_hops=max_hops,
        beam_width=max(DEFAULT_BEAM_WIDTH, top_paths * 5),
    )
    ranked_paths = _rank_path_candidates(
        query,
        candidates,
        chunks_by_id,
        nodes_by_id,
        edges_by_id,
        seed_scores,
        mention_scores,
    )

    graph_paths = []
    for candidate, score in ranked_paths:
        chunk_ids = _candidate_chunk_ids(
            candidate,
            chunks_by_id,
            nodes_by_id,
            edges_by_id,
        )
        if not chunk_ids:
            continue
        graph_paths.append(
            GraphPathHit(
                score=score,
                nodes=[nodes_by_id[node_id] for node_id in candidate.node_ids],
                edges=[edges_by_id[edge_id] for edge_id in candidate.edge_ids],
                chunk_ids=chunk_ids,
            )
        )
        if len(graph_paths) == top_paths:
            break

    ranked_chunks = _rank_evidence_chunks(graph_paths, chunks_by_id)[:top_chunks]
    index = _build_chunk_graph_index(graph, chunks_by_id, nodes_by_id)
    chunk_hits = _attach_graph_annotations(ranked_chunks, index, nodes_by_id)
    sources = _unique_ordered(hit.chunk.source for hit in chunk_hits)

    return RetrievalResult(
        query=query,
        sources=sources,
        references=_source_references(chunk_hits),
        chunks=chunk_hits,
        graph_paths=graph_paths,
        context_text=_format_context(query, chunk_hits, graph_paths),
    )


def _seed_node_scores(
    query: str,
    nodes: Sequence[Node],
    mention_scores: Mapping[str, float],
    embedding_client: EmbeddingClient | None,
    node_embedding_store: NodeEmbeddingStore | None,
) -> dict[str, float]:
    label_scores = _normalized_bm25_scores(
        query,
        {node.id: node.label for node in nodes},
    )
    metadata_scores = _normalized_bm25_scores(
        query,
        {node.id: _node_search_text(node) for node in nodes},
    )
    semantic_scores = _semantic_node_scores(
        query,
        nodes,
        embedding_client,
        node_embedding_store,
    )
    return _normalize_scores(
        {
            node.id: (
                label_scores.get(node.id, 0.0) * 2.0
                + mention_scores.get(node.id, 0.0) * 2.0
                + metadata_scores.get(node.id, 0.0)
                + semantic_scores.get(node.id, 0.0)
            )
            for node in nodes
        }
    )


def _mentioned_node_scores(
    query: str,
    nodes: Sequence[Node],
) -> dict[str, float]:
    query_tokens = tokenize(query)
    matches: list[tuple[str, int, int]] = []
    for node in nodes:
        for name in [node.label, *node.aliases]:
            name_tokens = tokenize(name)
            if not name_tokens:
                continue
            for start in range(len(query_tokens) - len(name_tokens) + 1):
                if query_tokens[start : start + len(name_tokens)] == name_tokens:
                    matches.append((node.id, start, len(name_tokens)))

    scores = {}
    for node_id, start, length in matches:
        end = start + length
        contained_by_longer_match = any(
            other_start <= start
            and end <= other_start + other_length
            and other_length > length
            for _, other_start, other_length in matches
        )
        if not contained_by_longer_match:
            scores[node_id] = 1.0
    return scores


def _node_search_text(node: Node) -> str:
    parts = [node.label, node.type, *node.aliases]
    if node.suggested_type:
        parts.append(node.suggested_type)
    if node.properties:
        parts.append(json.dumps(node.properties, sort_keys=True))
    return "\n".join(parts)


def _semantic_node_scores(
    query: str,
    nodes: Sequence[Node],
    embedding_client: EmbeddingClient | None,
    node_embedding_store: NodeEmbeddingStore | None,
) -> dict[str, float]:
    if embedding_client is None or node_embedding_store is None:
        return {}

    records = node_embedding_store.load()
    matching_records = {
        node.id: records[node.id]
        for node in nodes
        if node.id in records
        and records[node.id].embedding_model == embedding_client.embedding_model
    }
    if not matching_records:
        return {}

    query_vector = embedding_client.embed_texts([query])[0]
    return _normalize_scores(
        {
            node_id: _cosine_similarity(query_vector, record.vector)
            for node_id, record in matching_records.items()
        }
    )


def _traverse_paths(
    query: str,
    graph: KnowledgeGraph,
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
    seed_scores: Mapping[str, float],
    mention_scores: Mapping[str, float],
    *,
    max_hops: int,
    beam_width: int,
) -> list[_PathCandidate]:
    if not seed_scores:
        return []

    adjacency = _build_adjacency(graph)
    seed_ids = sorted(
        seed_scores,
        key=lambda node_id: (-seed_scores[node_id], node_id),
    )[:DEFAULT_TOP_SEEDS]
    frontier = [
        _PathCandidate(
            node_ids=(node_id,),
            edge_ids=(),
            seed_score=seed_scores[node_id],
        )
        for node_id in seed_ids
    ]
    candidates = []

    for _ in range(max_hops):
        expanded = []
        for candidate in frontier:
            current_node_id = candidate.node_ids[-1]
            for edge, neighbor_id in adjacency.get(current_node_id, []):
                if neighbor_id in candidate.node_ids:
                    continue
                expanded.append(
                    _PathCandidate(
                        node_ids=(*candidate.node_ids, neighbor_id),
                        edge_ids=(*candidate.edge_ids, edge.id),
                        seed_score=candidate.seed_score,
                    )
                )

        if not expanded:
            break

        expanded = _deduplicate_candidates(expanded)
        ranked = _rank_path_candidates(
            query,
            expanded,
            chunks_by_id,
            nodes_by_id,
            edges_by_id,
            seed_scores,
            mention_scores,
        )
        frontier = [candidate for candidate, _ in ranked[:beam_width]]
        candidates.extend(frontier)

    return _deduplicate_candidates(candidates)


def _build_adjacency(graph: KnowledgeGraph) -> dict[str, list[tuple[Edge, str]]]:
    adjacency: dict[str, list[tuple[Edge, str]]] = {}
    for edge in sorted(graph.edges, key=lambda item: item.id):
        adjacency.setdefault(edge.source, []).append((edge, edge.target))
        if edge.target != edge.source:
            adjacency.setdefault(edge.target, []).append((edge, edge.source))
    return adjacency


def _rank_path_candidates(
    query: str,
    candidates: Sequence[_PathCandidate],
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
    node_scores: Mapping[str, float],
    mention_scores: Mapping[str, float],
) -> list[tuple[_PathCandidate, float]]:
    if not candidates:
        return []

    path_documents = [
        BM25Document(
            id=str(index),
            text=_candidate_path_text(
                candidate,
                nodes_by_id,
                edges_by_id,
            ),
        )
        for index, candidate in enumerate(candidates)
    ]
    path_scores = _normalize_scores(
        {
            document.id: score
            for document, score in BM25Index(path_documents).rank(query)
        }
    )
    evidence_documents = [
        BM25Document(
            id=str(index),
            text=_candidate_evidence_text(
                candidate,
                chunks_by_id,
                nodes_by_id,
                edges_by_id,
            ),
        )
        for index, candidate in enumerate(candidates)
    ]
    evidence_scores = _normalize_scores(
        {
            document.id: score
            for document, score in BM25Index(evidence_documents).rank(query)
        }
    )
    scores = {
        str(index): (
            path_scores.get(str(index), 0.0)
            + evidence_scores.get(str(index), 0.0) * 0.5
            + _path_node_coverage(candidate, node_scores)
            + sum(
                mention_scores.get(node_id, 0.0)
                for node_id in candidate.node_ids
            )
        )
        * PATH_HOP_DECAY ** (len(candidate.edge_ids) - 1)
        for index, candidate in enumerate(candidates)
    }
    normalized_scores = _normalize_scores(scores)
    ranked = [
        (candidate, normalized_scores.get(str(index), 0.0))
        for index, candidate in enumerate(candidates)
    ]
    ranked.sort(
        key=lambda item: (
            -item[1],
            len(item[0].edge_ids),
            item[0].node_ids,
            item[0].edge_ids,
        )
    )
    return ranked


def _candidate_path_text(
    candidate: _PathCandidate,
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
) -> str:
    lines = []
    for index, edge_id in enumerate(candidate.edge_ids):
        edge = edges_by_id[edge_id]
        left = nodes_by_id[candidate.node_ids[index]]
        right = nodes_by_id[candidate.node_ids[index + 1]]
        direction = edge.label if edge.source == left.id else f"inverse {edge.label}"
        lines.append(
            f"{_node_search_text(left)}\n"
            f"{direction}\n"
            f"{_node_search_text(right)}"
        )

    return "\n".join(lines)


def _candidate_evidence_text(
    candidate: _PathCandidate,
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
) -> str:
    chunk_ids = _candidate_chunk_ids(
        candidate,
        chunks_by_id,
        nodes_by_id,
        edges_by_id,
    )
    return "\n".join(chunks_by_id[chunk_id].text for chunk_id in chunk_ids)


def _path_node_coverage(
    candidate: _PathCandidate,
    node_scores: Mapping[str, float],
) -> float:
    strongest_scores = sorted(
        (node_scores.get(node_id, 0.0) for node_id in candidate.node_ids),
        reverse=True,
    )[:2]
    return sum(strongest_scores) / 2.0


def _candidate_chunk_ids(
    candidate: _PathCandidate,
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
) -> list[str]:
    edge_chunk_ids = _unique_ordered(
        chunk_id
        for edge_id in candidate.edge_ids
        for chunk_id in edges_by_id[edge_id].chunk_ids
        if chunk_id in chunks_by_id
    )
    if edge_chunk_ids:
        return edge_chunk_ids
    return _unique_ordered(
        chunk_id
        for node_id in candidate.node_ids
        for chunk_id in nodes_by_id[node_id].chunk_ids
        if chunk_id in chunks_by_id
    )


def _deduplicate_candidates(
    candidates: Sequence[_PathCandidate],
) -> list[_PathCandidate]:
    by_path: dict[tuple[str, ...], _PathCandidate] = {}
    for candidate in candidates:
        reverse_edge_ids = tuple(reversed(candidate.edge_ids))
        key = min(candidate.edge_ids, reverse_edge_ids)
        existing = by_path.get(key)
        if existing is None or (
            candidate.seed_score,
            tuple(reversed(candidate.node_ids)),
        ) > (
            existing.seed_score,
            tuple(reversed(existing.node_ids)),
        ):
            by_path[key] = candidate
    return list(by_path.values())


def _rank_evidence_chunks(
    paths: Sequence[GraphPathHit],
    chunks_by_id: Mapping[str, Chunk],
) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    for rank, path in enumerate(paths, start=1):
        for chunk_id in path.chunk_ids:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / rank

    normalized_scores = _normalize_scores(scores)
    ranked = [
        (chunks_by_id[chunk_id], score)
        for chunk_id, score in normalized_scores.items()
    ]
    ranked.sort(key=lambda item: (-item[1], item[0].index, item[0].id))
    return ranked


def _normalized_bm25_scores(
    query: str,
    text_by_id: Mapping[str, str],
) -> dict[str, float]:
    index = BM25Index(
        [BM25Document(id=item_id, text=text) for item_id, text in text_by_id.items()]
    )
    return _normalize_scores(
        {document.id: score for document, score in index.rank(query)}
    )


def _normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    positive_scores = {
        item_id: score
        for item_id, score in scores.items()
        if score > 0.0
    }
    if not positive_scores:
        return {}
    max_score = max(positive_scores.values())
    return {
        item_id: score / max_score
        for item_id, score in positive_scores.items()
    }
