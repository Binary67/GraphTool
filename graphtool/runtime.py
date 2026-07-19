from dataclasses import dataclass
from pathlib import Path

from graphtool.chunking import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.graph import (
    JsonChunkExtractionStore,
    JsonEmbeddingStore,
    JsonGraphEmbeddingStore,
    JsonGraphStore,
    JsonKnowledgeBaseStore,
    JsonTaxonomySuggestionStore,
    KnowledgeGraph,
)
from graphtool.llm import AzureOpenAIClient
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.retrieval import (
    JsonChunkEmbeddingStore,
    RetrievalResult,
)
from graphtool.retrieval.hybrid_retriever import retrieve_hybrid_context

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_LOG_FILES = 3


@dataclass(frozen=True)
class GraphToolPaths:
    root: Path
    documents_dir: Path
    pdf_conversions_dir: Path
    chunks_dir: Path
    chunk_extractions_dir: Path
    graphs_dir: Path
    graph_embeddings_dir: Path
    chunk_embeddings_path: Path
    knowledge_base_path: Path
    knowledge_base_embeddings_path: Path
    taxonomy_suggestions_path: Path
    dropped_edges_path: Path
    logs_dir: Path
    visualizations_dir: Path


@dataclass(frozen=True)
class GraphToolRuntime:
    paths: GraphToolPaths
    graph_store: JsonGraphStore
    knowledge_base_store: JsonKnowledgeBaseStore
    graph_embedding_store: JsonGraphEmbeddingStore
    knowledge_base_embedding_store: JsonEmbeddingStore
    taxonomy_suggestion_store: JsonTaxonomySuggestionStore
    chunk_store: JsonChunkStore
    chunk_extraction_store: JsonChunkExtractionStore
    chunk_embedding_store: JsonChunkEmbeddingStore
    fast_llm: AzureOpenAIClient

    def search(self, query: str, *, top_chunks: int = 5) -> RetrievalResult:
        graph, chunks = self._search_inputs()
        return retrieve_hybrid_context(
            query,
            graph,
            chunks,
            top_chunks=top_chunks,
            embedding_client=self.fast_llm,
            chunk_embedding_store=self.chunk_embedding_store,
            node_embedding_store=self.knowledge_base_embedding_store,
        )

    def _search_inputs(self) -> tuple[KnowledgeGraph, list[Chunk]]:
        if not self.knowledge_base_store.exists():
            raise FileNotFoundError(
                "Knowledge base not found. Synchronize documents before searching."
            )
        return self.knowledge_base_store.load(), self.chunk_store.load_all()


def default_paths(root: str | Path | None = None) -> GraphToolPaths:
    project_root = Path(root) if root is not None else DEFAULT_PROJECT_ROOT
    data_dir = project_root / "data"
    return GraphToolPaths(
        root=project_root,
        documents_dir=project_root / "documents",
        pdf_conversions_dir=data_dir / "pdf_conversions",
        chunks_dir=data_dir / "chunks",
        chunk_extractions_dir=data_dir / "chunk_extractions",
        graphs_dir=data_dir / "graphs",
        graph_embeddings_dir=data_dir / "graph_embeddings",
        chunk_embeddings_path=data_dir / "chunk_embeddings.json",
        knowledge_base_path=data_dir / "knowledge_base.json",
        knowledge_base_embeddings_path=data_dir / "knowledge_base_embeddings.json",
        taxonomy_suggestions_path=data_dir / "taxonomy_suggestions.json",
        dropped_edges_path=data_dir / "dropped_edges.jsonl",
        logs_dir=project_root / "logs",
        visualizations_dir=data_dir / "visualizations",
    )


def create_runtime(
    config: AzureOpenAIConfig,
    *,
    paths: GraphToolPaths | None = None,
) -> GraphToolRuntime:
    runtime_paths = paths or default_paths()
    return GraphToolRuntime(
        paths=runtime_paths,
        graph_store=JsonGraphStore(runtime_paths.graphs_dir),
        knowledge_base_store=JsonKnowledgeBaseStore(runtime_paths.knowledge_base_path),
        graph_embedding_store=JsonGraphEmbeddingStore(
            runtime_paths.graph_embeddings_dir
        ),
        knowledge_base_embedding_store=JsonEmbeddingStore(
            runtime_paths.knowledge_base_embeddings_path
        ),
        taxonomy_suggestion_store=JsonTaxonomySuggestionStore(
            runtime_paths.taxonomy_suggestions_path
        ),
        chunk_store=JsonChunkStore(runtime_paths.chunks_dir),
        chunk_extraction_store=JsonChunkExtractionStore(
            runtime_paths.chunk_extractions_dir
        ),
        chunk_embedding_store=JsonChunkEmbeddingStore(
            runtime_paths.chunk_embeddings_path
        ),
        fast_llm=AzureOpenAIClient(
            config,
            text_deployment=config.fast_deployment,
        ),
    )
