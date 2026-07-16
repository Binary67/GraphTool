from collections.abc import Sequence

from graphtool.chunking.types import Chunk
from graphtool.graph.types import KnowledgeGraph
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.embedding_store import ChunkEmbeddingStore
from graphtool.retrieval.graph_retriever import (
    DEFAULT_MAX_HOPS,
    DEFAULT_TOP_PATHS,
    NodeEmbeddingStore,
    retrieve_graph_context,
)
from graphtool.retrieval.retriever import (
    _format_context,
    _source_references,
    _unique_ordered,
    retrieve_context,
)
from graphtool.retrieval.types import ChunkHit, RetrievalResult

RECIPROCAL_RANK_CONSTANT = 60


def retrieve_hybrid_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    node_embedding_store: NodeEmbeddingStore | None = None,
) -> RetrievalResult:
    direct_result = retrieve_context(
        query,
        graph,
        chunks,
        top_chunks=top_chunks,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
    )
    graph_result = retrieve_graph_context(
        query,
        graph,
        chunks,
        max_hops=max_hops,
        top_paths=top_paths,
        top_chunks=top_chunks,
        embedding_client=embedding_client,
        node_embedding_store=node_embedding_store,
    )
    chunk_hits = _fuse_chunk_hits(
        direct_result.chunks,
        graph_result.chunks,
    )[:top_chunks]
    sources = _unique_ordered(hit.chunk.source for hit in chunk_hits)

    return RetrievalResult(
        query=query,
        sources=sources,
        references=_source_references(chunk_hits),
        chunks=chunk_hits,
        graph_paths=graph_result.graph_paths,
        context_text=_format_context(
            query,
            chunk_hits,
            graph_result.graph_paths,
        ),
    )


def _fuse_chunk_hits(
    direct_hits: Sequence[ChunkHit],
    graph_hits: Sequence[ChunkHit],
) -> list[ChunkHit]:
    hits_by_id = {
        hit.chunk.id: hit
        for hit in [*direct_hits, *graph_hits]
    }
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for hits in (direct_hits, graph_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + (
                1.0 / (RECIPROCAL_RANK_CONSTANT + rank)
            )
            best_rank[hit.chunk.id] = min(best_rank.get(hit.chunk.id, rank), rank)

    ranked = [
        hit.model_copy(update={"score": scores[chunk_id]})
        for chunk_id, hit in hits_by_id.items()
    ]
    ranked.sort(
        key=lambda hit: (
            -hit.score,
            best_rank[hit.chunk.id],
            hit.chunk.index,
            hit.chunk.id,
        )
    )
    return ranked
