import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.markdown import chunk_markdown
from graphtool.chunking.types import Chunk
from graphtool.graph.embedding_store import (
    JsonEmbeddingStore,
    JsonGraphEmbeddingStore,
    NodeEmbeddingRecord,
)
from graphtool.graph.extraction_store import JsonChunkExtractionStore
from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.generator import generate_knowledge_graph
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.provenance import remove_source_from_knowledge_graph
from graphtool.graph.resolver import (
    DEFAULT_MIN_CANDIDATE_SIMILARITY,
    SemanticEntityResolver,
)
from graphtool.graph.taxonomy import (
    JsonTaxonomySuggestionStore,
    TaxonomySuggestionRecord,
)
from graphtool.graph.types import KnowledgeGraph
from graphtool.llm.base import LLMClient
from graphtool.retrieval.embedding_store import ChunkEmbeddingStore
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import document_content_hash

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class CorpusSyncResult:
    added_sources: list[str]
    changed_sources: list[str]
    deleted_sources: list[str]
    unchanged_sources: list[str]


@dataclass(frozen=True)
class _PreparedDocument:
    source: str
    chunks: list[Chunk]
    graph: KnowledgeGraph
    taxonomy_suggestions: list[TaxonomySuggestionRecord]


@dataclass
class _TaxonomySuggestionBuffer:
    records: list[TaxonomySuggestionRecord] = field(default_factory=list)

    def append_many(self, records: Sequence[TaxonomySuggestionRecord]) -> None:
        self.records.extend(records)


def synchronize_documents(
    documents: Mapping[str, str],
    graph_store: JsonGraphStore,
    chunk_store: JsonChunkStore,
    llm: LLMClient,
    *,
    knowledge_base_store: JsonKnowledgeBaseStore,
    graph_embedding_store: JsonGraphEmbeddingStore | None = None,
    knowledge_base_embedding_store: JsonEmbeddingStore | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    dropped_edges_path: Path | None = None,
    taxonomy_suggestion_store: JsonTaxonomySuggestionStore | None = None,
    chunk_extraction_store: JsonChunkExtractionStore | None = None,
    min_candidate_similarity: float = DEFAULT_MIN_CANDIDATE_SIMILARITY,
    chunk_generation_workers: int = 4,
) -> CorpusSyncResult:
    if chunk_generation_workers < 1:
        raise ValueError("chunk_generation_workers must be positive")

    existing_graphs = graph_store.load_all()
    existing_by_source = {}
    for graph in existing_graphs:
        if graph.metadata is None:
            raise ValueError("Cannot synchronize graph without metadata.source.")
        existing_by_source[graph.metadata.source] = graph

    content_hashes = {
        source: document_content_hash(markdown)
        for source, markdown in documents.items()
    }
    current_sources = set(documents)
    existing_sources = set(existing_by_source)
    added_sources = sorted(current_sources - existing_sources)
    deleted_sources = sorted(existing_sources - current_sources)
    changed_sources = sorted(
        source
        for source in current_sources & existing_sources
        if existing_by_source[source].metadata.content_hash != content_hashes[source]
    )
    unchanged_sources = sorted(
        current_sources - set(added_sources) - set(changed_sources)
    )
    RUN_LOGGER.info(
        "Knowledge graph changes: %s added, %s changed, %s removed, %s unchanged",
        len(added_sources),
        len(changed_sources),
        len(deleted_sources),
        len(unchanged_sources),
    )

    prepared = []
    sources_to_prepare = [*added_sources, *changed_sources]
    for index, source in enumerate(sources_to_prepare, start=1):
        markdown = documents[source]
        chunks = chunk_markdown(markdown, source)
        RUN_LOGGER.info(
            "[%s/%s] Building knowledge graph: %s (%s %s)",
            index,
            len(sources_to_prepare),
            source,
            len(chunks),
            "chunk" if len(chunks) == 1 else "chunks",
        )
        resolver = _make_semantic_resolver(
            llm,
            graph_embedding_store,
            source=source,
            min_candidate_similarity=min_candidate_similarity,
        )
        suggestion_buffer = (
            _TaxonomySuggestionBuffer()
            if taxonomy_suggestion_store is not None
            else None
        )
        graph = generate_knowledge_graph(
            chunks,
            source,
            llm,
            content_hash=content_hashes[source],
            resolver=resolver,
            dropped_edges_path=dropped_edges_path,
            taxonomy_suggestion_store=suggestion_buffer,
            extraction_store=chunk_extraction_store,
            max_workers=chunk_generation_workers,
        )
        RUN_LOGGER.info(
            "Built knowledge graph: %s (%s %s, %s %s)",
            source,
            len(graph.nodes),
            "entity" if len(graph.nodes) == 1 else "entities",
            len(graph.edges),
            "relationship" if len(graph.edges) == 1 else "relationships",
        )
        prepared.append(
            _PreparedDocument(
                source=source,
                chunks=chunks,
                graph=graph,
                taxonomy_suggestions=(
                    list(suggestion_buffer.records)
                    if suggestion_buffer is not None
                    else []
                ),
            )
        )

    result = CorpusSyncResult(
        added_sources=added_sources,
        changed_sources=changed_sources,
        deleted_sources=deleted_sources,
        unchanged_sources=unchanged_sources,
    )
    if not prepared and not deleted_sources and knowledge_base_store.exists():
        return result

    removed_sources = [*changed_sources, *deleted_sources]
    old_chunk_ids = [
        chunk.id
        for source in removed_sources
        for chunk in chunk_store.load(source)
    ]

    reusable_embedding_sources = (
        sources_to_prepare
        if knowledge_base_store.exists()
        else sorted(current_sources)
    )
    resolver = _make_semantic_resolver(
        llm,
        knowledge_base_embedding_store,
        min_candidate_similarity=min_candidate_similarity,
        reusable_embedding_records=_load_document_embedding_records(
            graph_embedding_store,
            reusable_embedding_sources,
        ),
    )
    if knowledge_base_store.exists():
        knowledge_base = knowledge_base_store.load()
        for source in removed_sources:
            knowledge_base = remove_source_from_knowledge_graph(
                knowledge_base,
                source,
            )
        if resolver is not None:
            knowledge_base = resolver.combine_into(
                knowledge_base,
                [item.graph for item in prepared],
            )
        else:
            knowledge_base = combine_knowledge_graphs(
                [knowledge_base, *(item.graph for item in prepared)]
            )
    else:
        final_graphs = [
            graph
            for source, graph in existing_by_source.items()
            if source not in removed_sources
        ]
        final_graphs.extend(item.graph for item in prepared)
        knowledge_base = _combine_knowledge_graphs(final_graphs, resolver)

    for source in deleted_sources:
        graph_store.delete(source)
        chunk_store.delete(source)
        if graph_embedding_store is not None:
            graph_embedding_store.delete(source)
        if chunk_extraction_store is not None:
            chunk_extraction_store.delete(source)
        if taxonomy_suggestion_store is not None:
            taxonomy_suggestion_store.delete_source(source)

    for item in prepared:
        chunk_store.save(item.source, item.chunks)
        graph_store.save(item.graph)
        if taxonomy_suggestion_store is not None:
            taxonomy_suggestion_store.replace_source(
                item.source,
                item.taxonomy_suggestions,
            )

    if chunk_embedding_store is not None and old_chunk_ids:
        chunk_embedding_store.delete(old_chunk_ids)

    if prepared or deleted_sources or not knowledge_base_store.exists():
        knowledge_base_store.save(knowledge_base)

    return result


def rebuild_knowledge_base(
    graph_store: JsonGraphStore,
    knowledge_base_store: JsonKnowledgeBaseStore,
    *,
    resolver: SemanticEntityResolver | None = None,
) -> KnowledgeGraph:
    graph = _combine_knowledge_graphs(graph_store.load_all(), resolver)
    knowledge_base_store.save(graph)
    return graph


def _combine_knowledge_graphs(
    graphs: list[KnowledgeGraph],
    resolver: SemanticEntityResolver | None,
) -> KnowledgeGraph:
    if resolver is None:
        return combine_knowledge_graphs(graphs)
    return resolver.combine_into(None, graphs)


def _make_semantic_resolver(
    llm: LLMClient,
    graph_embedding_store: JsonGraphEmbeddingStore | JsonEmbeddingStore | None,
    *,
    source: str | None = None,
    min_candidate_similarity: float = DEFAULT_MIN_CANDIDATE_SIMILARITY,
    reusable_embedding_records: Sequence[NodeEmbeddingRecord] = (),
) -> SemanticEntityResolver | None:
    if (
        not hasattr(llm, "embed_texts")
        or not hasattr(llm, "embedding_model")
    ):
        return None

    embedding_store = None
    if isinstance(graph_embedding_store, JsonGraphEmbeddingStore):
        if source is None:
            raise ValueError("source is required for per-document graph embeddings.")
        embedding_store = _SourceEmbeddingStore(graph_embedding_store, source)
    else:
        embedding_store = graph_embedding_store

    return SemanticEntityResolver(
        llm,
        llm,
        embedding_store,
        min_candidate_similarity=min_candidate_similarity,
        reusable_embedding_records=reusable_embedding_records,
    )


def _load_document_embedding_records(
    store: JsonGraphEmbeddingStore | None,
    sources: Sequence[str],
) -> list[NodeEmbeddingRecord]:
    if store is None:
        return []
    return [
        record
        for source in sources
        if store.exists(source)
        for record in store.load(source).values()
    ]


class _SourceEmbeddingStore:
    def __init__(self, store: JsonGraphEmbeddingStore, source: str) -> None:
        self._store = store
        self._source = source

    def load(self):
        return self._store.load(self._source)

    def save(self, records):
        self._store.save(self._source, records)
