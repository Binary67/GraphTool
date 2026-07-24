from dataclasses import dataclass, field
from pathlib import Path

from graphtool.chunking import SqliteChunkStore
from graphtool.chunking.types import Chunk
from graphtool.corpus_stores import SqliteCorpusStores
from graphtool.graph import (
    JsonChunkExtractionStore,
    KnowledgeGraph,
    SqliteEmbeddingStore,
    SqliteGraphStore,
    SqliteGraphEmbeddingStore,
    SqliteKnowledgeBaseStore,
    SqliteTaxonomySuggestionStore,
)
from graphtool.llm import AzureOpenAIAudioTranscriber, AzureOpenAIClient
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.retrieval import (
    RetrievalResult,
    SqliteChunkEmbeddingStore,
)
from graphtool.retrieval.hybrid_retriever import (
    PreparedHybridRetriever,
    prepare_hybrid_retriever,
)
from graphtool.storage import open_database

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_LOG_FILES = 3


@dataclass(frozen=True)
class GraphToolPaths:
    root: Path
    documents_dir: Path
    audio_transcriptions_dir: Path
    audio_transcription_glossary_path: Path
    pdf_conversions_dir: Path
    presentation_conversions_dir: Path
    chunk_extractions_dir: Path
    db_path: Path
    dropped_edges_path: Path
    logs_dir: Path
    visualizations_dir: Path


@dataclass
class GraphToolRuntime:
    paths: GraphToolPaths
    corpus_stores: SqliteCorpusStores
    chunk_extraction_store: JsonChunkExtractionStore
    fast_llm: AzureOpenAIClient
    audio_transcriber: AzureOpenAIAudioTranscriber
    _search_retriever: PreparedHybridRetriever | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def search(self, query: str, *, top_chunks: int = 5) -> RetrievalResult:
        if self._search_retriever is None:
            raise RuntimeError(
                "Search is not prepared. Call prepare_search after synchronization."
            )
        return self._search_retriever.retrieve(query, top_chunks=top_chunks)

    @property
    def graph_store(self) -> SqliteGraphStore:
        return self.corpus_stores.graphs

    @property
    def knowledge_base_store(self) -> SqliteKnowledgeBaseStore:
        return self.corpus_stores.knowledge_base

    @property
    def graph_embedding_store(self) -> SqliteGraphEmbeddingStore:
        return self.corpus_stores.graph_embeddings

    @property
    def knowledge_base_embedding_store(self) -> SqliteEmbeddingStore:
        return self.corpus_stores.knowledge_base_embeddings

    @property
    def taxonomy_suggestion_store(self) -> SqliteTaxonomySuggestionStore:
        return self.corpus_stores.taxonomy_suggestions

    @property
    def chunk_store(self) -> SqliteChunkStore:
        return self.corpus_stores.chunks

    @property
    def chunk_embedding_store(self) -> SqliteChunkEmbeddingStore:
        return self.corpus_stores.chunk_embeddings

    def prepare_search(self) -> None:
        graph, chunks = self._search_inputs()
        self._search_retriever = prepare_hybrid_retriever(
            graph,
            chunks,
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
        audio_transcriptions_dir=data_dir / "audio_transcriptions",
        audio_transcription_glossary_path=(
            project_root / "config" / "transcription_glossary.json"
        ),
        pdf_conversions_dir=data_dir / "pdf_conversions",
        presentation_conversions_dir=data_dir / "presentation_conversions",
        chunk_extractions_dir=data_dir / "chunk_extractions",
        db_path=data_dir / "graphtool.db",
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
    conn = open_database(runtime_paths.db_path)
    return GraphToolRuntime(
        paths=runtime_paths,
        corpus_stores=SqliteCorpusStores.from_connection(conn),
        chunk_extraction_store=JsonChunkExtractionStore(
            runtime_paths.chunk_extractions_dir
        ),
        fast_llm=AzureOpenAIClient(
            config,
            text_deployment=config.fast_deployment,
        ),
        audio_transcriber=AzureOpenAIAudioTranscriber(config),
    )
