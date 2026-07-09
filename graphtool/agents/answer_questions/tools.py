from langchain_core.tools import BaseTool, StructuredTool

from graphtool import corpus
from graphtool.agents.answer_questions.types import RetrievedContext
from graphtool.chunking.json_store import JsonChunkStore
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.embedding_store import ChunkEmbeddingStore


def make_retrieve_knowledge_context_tool(
    graph_store: JsonGraphStore,
    chunk_store: JsonChunkStore,
    *,
    knowledge_base_store: JsonKnowledgeBaseStore | None = None,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    top_nodes: int = 5,
    top_edges: int = 5,
    top_chunks: int = 5,
) -> BaseTool:
    def retrieve_knowledge_context(query: str) -> str:
        """Search the knowledge graph and return relevant context for a query."""
        result = corpus.search_knowledge_base(
            query,
            graph_store,
            chunk_store,
            knowledge_base_store=knowledge_base_store,
            embedding_client=embedding_client,
            chunk_embedding_store=chunk_embedding_store,
            top_nodes=top_nodes,
            top_edges=top_edges,
            top_chunks=top_chunks,
        )
        return RetrievedContext(
            query=result.query,
            sources=result.sources,
            context_text=result.context_text,
        ).model_dump_json()

    return StructuredTool.from_function(
        retrieve_knowledge_context,
        name="retrieve_knowledge_context",
        description=(
            "Search GraphTool's knowledge graph and document evidence. "
            "Input should be a focused natural-language search query. "
            "Returns JSON with query, sources, and context_text."
        ),
    )
