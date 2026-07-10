import json

from langchain_core.tools import BaseTool, StructuredTool

from graphtool import corpus
from graphtool.agents.answer_questions.types import (
    ChunkNeighborhood,
    ChunkReference,
    NeighborhoodChunk,
    RetrievedContext,
)
from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.embedding_store import ChunkEmbeddingStore
from graphtool.retrieval.retriever import retrieve_context

AllowedChunkKeys = set[tuple[str, str]]


def make_retrieve_knowledge_context_tool(
    graph_store: JsonGraphStore,
    chunk_store: JsonChunkStore,
    *,
    knowledge_base_store: JsonKnowledgeBaseStore | None = None,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    allowed_chunks: AllowedChunkKeys | None = None,
    top_chunks: int = 5,
) -> BaseTool:
    context = corpus.load_search_context(
        graph_store,
        chunk_store,
        knowledge_base_store=knowledge_base_store,
    )

    def retrieve_knowledge_context(query: str) -> str:
        """Search the knowledge graph and return relevant context for a query."""
        result = retrieve_context(
            query,
            context.graph,
            context.chunks,
            top_chunks=top_chunks,
            embedding_client=embedding_client,
            chunk_embedding_store=chunk_embedding_store,
        )
        chunk_references = [
            ChunkReference(
                chunk_id=hit.chunk.id,
                source=hit.chunk.source,
                index=hit.chunk.index,
                heading_path=hit.chunk.heading_path,
            )
            for hit in result.chunks
        ]
        if allowed_chunks is not None:
            allowed_chunks.update(
                (ref.source, ref.chunk_id) for ref in chunk_references
            )
        return RetrievedContext(
            query=result.query,
            sources=result.sources,
            chunk_references=chunk_references,
            context_text=result.context_text,
        ).model_dump_json()

    return StructuredTool.from_function(
        retrieve_knowledge_context,
        name="retrieve_knowledge_context",
        description=(
            "Search GraphTool's knowledge graph and document evidence. This must "
            "be the first retrieval tool used for a question. "
            "Input should be a focused natural-language search query. "
            "Returns typed JSON with source paths, structured chunk references, "
            "and evidence text."
        ),
    )


def get_chunk_neighborhood(
    chunk_store: JsonChunkStore,
    source: str,
    chunk_id: str,
) -> ChunkNeighborhood:
    previous, current, next_chunk = chunk_store.load_neighborhood(source, chunk_id)
    return ChunkNeighborhood(
        source=source,
        chunk_id=chunk_id,
        previous=(
            _neighborhood_chunk(previous)
            if previous is not None
            else None
        ),
        current=_neighborhood_chunk(current),
        next=(
            _neighborhood_chunk(next_chunk)
            if next_chunk is not None
            else None
        ),
    )


def make_get_chunk_neighborhood_tool(
    chunk_store: JsonChunkStore,
    allowed_chunks: AllowedChunkKeys | None = None,
) -> BaseTool:
    def lookup(source: str, chunk_id: str) -> str:
        """Return the chunks immediately before and after a searched chunk."""
        if allowed_chunks is not None and (source, chunk_id) not in allowed_chunks:
            return json.dumps(
                {
                    "error": (
                        f"Unknown chunk_id {chunk_id!r} for source {source!r}. "
                        "Use a source and chunk_id pair from a prior "
                        "retrieve_knowledge_context result."
                    )
                }
            )
        return get_chunk_neighborhood(
            chunk_store,
            source,
            chunk_id,
        ).model_dump_json()

    return StructuredTool.from_function(
        lookup,
        name="get_chunk_neighborhood",
        description=(
            "Return the previous, current, and next chunks from one source as "
            "typed JSON. Use only a source and chunk_id pair from a prior "
            "retrieve_knowledge_context result, and only when that passage is "
            "incomplete or needs adjacent document context."
        ),
    )


def _neighborhood_chunk(chunk: Chunk) -> NeighborhoodChunk:
    return NeighborhoodChunk(
        chunk_id=chunk.id,
        source=chunk.source,
        index=chunk.index,
        heading_path=chunk.heading_path,
        text=chunk.text,
    )
