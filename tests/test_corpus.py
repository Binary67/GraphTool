from datetime import datetime, timezone
from typing import TypeVar

import pytest

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.corpus import (
    load_markdown_documents,
    load_search_context,
    rebuild_knowledge_base,
    search_knowledge_base,
    synchronize_documents,
)
from graphtool.graph.generator import (
    _ExtractedEdge,
    _ExtractedKnowledgeGraph,
    _ExtractedNode,
)
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.embedding_store import JsonEmbeddingStore, JsonGraphEmbeddingStore
from graphtool.graph.resolver import EntityResolutionDecision
from graphtool.graph.taxonomy import (
    JsonTaxonomySuggestionStore,
    TaxonomySuggestionRecord,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage
from graphtool.retrieval import ChunkEmbeddingRecord, JsonChunkEmbeddingStore
from graphtool.source import document_content_hash, source_key

T = TypeVar("T")


class FakeLLM:
    def __init__(self, responses: list[_ExtractedKnowledgeGraph]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return self.responses[len(self.calls) - 1]


class FailingLLM:
    def generate_structured(self, messages, response_model):
        raise RuntimeError("extraction failed")


class FakeSemanticLLM:
    embedding_model = "fake-embedding-model"

    def __init__(
        self,
        responses: list[_ExtractedKnowledgeGraph],
        decisions: list[EntityResolutionDecision],
        vectors: dict[str, list[float]],
    ) -> None:
        self.responses = responses
        self.decisions = decisions
        self.vectors = vectors
        self.calls: list[tuple[list[LLMMessage], type]] = []
        self.embedding_calls: list[str] = []
        self.embedding_batch_calls: list[list[str]] = []
        self._response_index = 0
        self._decision_index = 0

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        if response_model is EntityResolutionDecision:
            decision = self.decisions[self._decision_index]
            self._decision_index += 1
            return decision

        response = self.responses[self._response_index]
        self._response_index += 1
        return response

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.embedding_batch_calls.append(batch)
        self.embedding_calls.extend(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
        label_line = text.splitlines()[0] if text else ""
        for key, vector in self.vectors.items():
            if key in label_line:
                return vector
        return [0.0, 1.0]


class FakeEmbeddingClient:
    embedding_model = "fake-embedding-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.calls.extend(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
        for marker, vector in self.vectors.items():
            if marker in text:
                return vector
        return [0.0, 1.0]


def _chunk(source: str, text: str, heading: str) -> Chunk:
    return Chunk(
        id=f"{source_key(source)}-chunk-0000",
        source=source,
        index=0,
        text=text,
        heading_path=[heading],
    )


def _extracted_graph(
    nodes: list[_ExtractedNode],
    edges: list[_ExtractedEdge] | None = None,
) -> _ExtractedKnowledgeGraph:
    return _ExtractedKnowledgeGraph(nodes=nodes, edges=edges or [])


def _graph(source: str, chunk: Chunk, node_id: str, label: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id=node_id,
                label=label,
                type="Concept",
                properties={"topic": "validation"},
                chunk_ids=[chunk.id],
            )
        ],
        edges=[],
        metadata=GraphMetadata(
            source=source,
            content_hash=document_content_hash(chunk.text),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def _relationship_graph(
    source: str,
    chunk_id: str,
    content: str = "# Existing\nGraphTool uses Azure OpenAI.",
) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="graphtool",
                label="GraphTool",
                type="Project",
                chunk_ids=[chunk_id],
            ),
            Node(
                id="azure-openai",
                label="Azure OpenAI",
                type="Service",
                chunk_ids=[chunk_id],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="graphtool",
                target="azure-openai",
                label="uses",
                chunk_ids=[chunk_id],
            )
        ],
        metadata=GraphMetadata(
            source=source,
            content_hash=document_content_hash(content),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_load_markdown_documents_returns_empty_for_missing_directory(tmp_path):
    documents = load_markdown_documents(tmp_path / "missing", source_root=tmp_path)

    assert documents == {}


def test_load_markdown_documents_reads_nested_markdown_relative_to_source_root(
    tmp_path,
):
    documents_dir = tmp_path / "documents"
    nested_dir = documents_dir / "guides"
    nested_dir.mkdir(parents=True)
    (documents_dir / "b.txt").write_text("ignored")
    (nested_dir / "z.md").write_text("# Z")
    (documents_dir / "a.md").write_text("# A")

    documents = load_markdown_documents(documents_dir, source_root=tmp_path)

    assert list(documents) == ["documents/a.md", "documents/guides/z.md"]
    assert documents == {
        "documents/a.md": "# A",
        "documents/guides/z.md": "# Z",
    }


def test_search_knowledge_base_searches_all_saved_documents(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    pydantic_chunk = _chunk(
        "docs/pydantic.md",
        "# Pydantic\nPydantic handles data validation.",
        "Pydantic",
    )
    fastapi_chunk = _chunk(
        "docs/fastapi.md",
        "# FastAPI\nFastAPI handles request validation.",
        "FastAPI",
    )
    chunk_store.save("docs/pydantic.md", [pydantic_chunk])
    chunk_store.save("docs/fastapi.md", [fastapi_chunk])
    graph_store.save(_graph("docs/pydantic.md", pydantic_chunk, "pydantic", "Pydantic"))
    graph_store.save(_graph("docs/fastapi.md", fastapi_chunk, "fastapi", "FastAPI"))

    result = search_knowledge_base("validation", graph_store, chunk_store)

    assert {hit.chunk.source for hit in result.chunks} == {
        "docs/pydantic.md",
        "docs/fastapi.md",
    }
    assert set(result.sources) == {"docs/pydantic.md", "docs/fastapi.md"}


def test_synchronize_documents_skips_unchanged_sources(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(_graph("docs/processed.md", chunk, "processed", "Processed"))
    rebuild_knowledge_base(graph_store, knowledge_base_store)
    fake = FakeLLM([])

    result = synchronize_documents(
        {"docs/processed.md": "# Processed\nText."},
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
    )

    assert result.unchanged_sources == ["docs/processed.md"]
    assert result.added_sources == []
    assert result.changed_sources == []
    assert result.deleted_sources == []
    assert fake.calls == []


def test_synchronize_documents_adds_only_pending_source(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    processed_chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(
        _graph("docs/processed.md", processed_chunk, "processed", "Processed")
    )
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[_ExtractedNode(ref="pending", label="Pending", type="concept")]
            )
        ]
    )

    result = synchronize_documents(
        {
            "docs/processed.md": "# Processed\nText.",
            "docs/pending.md": "# Pending\nNeeds validation.",
        },
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=JsonKnowledgeBaseStore(
            tmp_path / "knowledge_base.json"
        ),
    )

    assert result.added_sources == ["docs/pending.md"]
    assert result.unchanged_sources == ["docs/processed.md"]
    assert len(fake.calls) == 1
    assert graph_store.exists("docs/pending.md") is True
    assert chunk_store.load("docs/pending.md")


def test_synchronize_without_semantic_resolver_preserves_scoped_knowledge_base_nodes(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    existing_graph = _relationship_graph("docs/existing.md", "existing-chunk-0000")
    graph_store.save(existing_graph)
    rebuild_knowledge_base(graph_store, knowledge_base_store)
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="graphtool", label="GraphTool", type="tool"),
                    _ExtractedNode(
                        ref="azure-openai",
                        label="Azure OpenAI",
                        type="service",
                    ),
                ],
                edges=[
                    _ExtractedEdge(
                        id="llm-edge",
                        source_ref="graphtool",
                        target_ref="azure-openai",
                        label="uses",
                    )
                ],
            )
        ]
    )
    new_source = "docs/new.md"
    new_chunk_id = f"{source_key(new_source)}-chunk-0000"

    synchronize_documents(
        {
            "docs/existing.md": "# Existing\nGraphTool uses Azure OpenAI.",
            new_source: "# GraphTool\nUses Azure OpenAI.",
        },
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
    )

    graph = knowledge_base_store.load()
    assert {node.id for node in graph.nodes} == {
        "graphtool",
        "azure-openai",
        f"{new_chunk_id}::node-0001",
        f"{new_chunk_id}::node-0002",
    }
    assert {(edge.source, edge.target, edge.label) for edge in graph.edges} == {
        ("graphtool", "azure-openai", "uses"),
        (
            f"{new_chunk_id}::node-0001",
            f"{new_chunk_id}::node-0002",
            "uses",
        ),
    }


def test_synchronize_scopes_reused_node_refs_across_documents(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="node-1",
                        label="Anthropic",
                        type="organization",
                    )
                ]
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="node-1",
                        label="OpenAI",
                        type="organization",
                    )
                ]
            ),
        ]
    )
    first_source = "docs/openai.md"
    second_source = "docs/anthropic.md"
    first_chunk_id = f"{source_key(first_source)}-chunk-0000"
    second_chunk_id = f"{source_key(second_source)}-chunk-0000"

    synchronize_documents(
        {
            first_source: "# OpenAI\nAI company.",
            second_source: "# Anthropic\nAI company.",
        },
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
    )

    graph = knowledge_base_store.load()
    assert {node.id: node.label for node in graph.nodes} == {
        f"{first_chunk_id}::node-0001": "OpenAI",
        f"{second_chunk_id}::node-0001": "Anthropic",
    }


def test_synchronize_documents_updates_cached_knowledge_base_semantically(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    graph_embedding_store = JsonGraphEmbeddingStore(tmp_path / "graph_embeddings")
    knowledge_base_embedding_store = JsonEmbeddingStore(
        tmp_path / "knowledge_base_embeddings.json"
    )
    existing_chunk_id = "existing-chunk-0000"
    existing_graph = KnowledgeGraph(
        nodes=[
            Node(
                id="openai",
                label="OpenAI",
                type="Organization",
                chunk_ids=[existing_chunk_id],
            ),
            Node(
                id="chatgpt",
                label="ChatGPT",
                type="Product",
                chunk_ids=[existing_chunk_id],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="openai",
                target="chatgpt",
                label="develops",
                chunk_ids=[existing_chunk_id],
            )
        ],
        metadata=GraphMetadata(
            source="docs/existing.md",
            content_hash=document_content_hash("# Existing\nOpenAI develops ChatGPT."),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )
    graph_store.save(existing_graph)
    rebuild_knowledge_base(graph_store, knowledge_base_store)
    fake = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="openai-organization",
                        label="OpenAI organization",
                        type="organization",
                    ),
                    _ExtractedNode(
                        ref="chatgpt",
                        label="ChatGPT",
                        type="product",
                    ),
                ],
                edges=[
                    _ExtractedEdge(
                        id="new-edge",
                        source_ref="openai-organization",
                        target_ref="chatgpt",
                        label="develops",
                    )
                ],
            )
        ],
        decisions=[
            EntityResolutionDecision(
                decision="merge",
                target_node_id="openai",
                confidence=0.95,
                aliases_to_add=["OpenAI Inc."],
            )
        ],
        vectors={
            "OpenAI organization": [1.0, 0.0],
            "OpenAI": [1.0, 0.0],
            "ChatGPT": [0.0, 1.0],
        },
    )
    new_source = "docs/new.md"
    new_chunk_id = f"{source_key(new_source)}-chunk-0000"

    synchronize_documents(
        {
            "docs/existing.md": "# Existing\nOpenAI develops ChatGPT.",
            new_source: "# OpenAI organization\nDevelops ChatGPT.",
        },
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
        graph_embedding_store=graph_embedding_store,
        knowledge_base_embedding_store=knowledge_base_embedding_store,
    )

    graph = knowledge_base_store.load()
    assert {node.id for node in graph.nodes} == {"openai", "chatgpt"}
    openai = next(node for node in graph.nodes if node.id == "openai")
    assert openai.aliases == ["OpenAI organization", "OpenAI Inc."]
    assert openai.chunk_ids == [existing_chunk_id, new_chunk_id]
    assert openai.provenance[1].resolution_aliases == ["OpenAI Inc."]
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "openai"
    assert graph.edges[0].target == "chatgpt"
    assert graph.edges[0].chunk_ids == [existing_chunk_id, new_chunk_id]
    assert graph_embedding_store.exists(new_source) is True
    assert knowledge_base_embedding_store.exists() is True


def test_synchronize_documents_uses_min_candidate_similarity_for_resolvers(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    graph_embedding_store = JsonGraphEmbeddingStore(tmp_path / "graph_embeddings")
    knowledge_base_embedding_store = JsonEmbeddingStore(
        tmp_path / "knowledge_base_embeddings.json"
    )
    existing_graph = KnowledgeGraph(
        nodes=[
            Node(
                id="openai",
                label="OpenAI",
                type="Organization",
                chunk_ids=["existing-chunk-0000"],
            )
        ],
        edges=[],
        metadata=GraphMetadata(
            source="docs/existing.md",
            content_hash=document_content_hash("# Existing\nOpenAI."),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )
    graph_store.save(existing_graph)
    rebuild_knowledge_base(graph_store, knowledge_base_store)
    fake = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="openai-organization",
                        label="OpenAI organization",
                        type="organization",
                    )
                ]
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="openai-company",
                        label="OpenAI company",
                        type="organization",
                    )
                ]
            ),
        ],
        decisions=[],
        vectors={
            "OpenAI organization": [0.9, 0.1],
            "OpenAI company": [0.8, 0.2],
            "OpenAI": [1.0, 0.0],
        },
    )
    new_source = "docs/new.md"

    synchronize_documents(
        {
            "docs/existing.md": "# Existing\nOpenAI.",
            new_source: (
                "# OpenAI organization\nFirst mention.\n\n"
                "# OpenAI company\nSecond mention."
            )
        },
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
        graph_embedding_store=graph_embedding_store,
        knowledge_base_embedding_store=knowledge_base_embedding_store,
        min_candidate_similarity=1.0,
    )

    document_graph = graph_store.load(new_source)
    knowledge_base = knowledge_base_store.load()
    first_chunk_id = f"{source_key(new_source)}-chunk-0000"
    second_chunk_id = f"{source_key(new_source)}-chunk-0001"

    assert {node.id for node in document_graph.nodes} == {
        f"{first_chunk_id}::node-0001",
        f"{second_chunk_id}::node-0001",
    }
    assert {node.id for node in knowledge_base.nodes} == {
        "openai",
        f"{first_chunk_id}::node-0001",
        f"{second_chunk_id}::node-0001",
    }
    assert [
        response_model
        for _, response_model in fake.calls
        if response_model is EntityResolutionDecision
    ] == []


def test_synchronize_documents_rebuilds_missing_knowledge_base_cache(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    processed_chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(
        _graph("docs/processed.md", processed_chunk, "processed", "Processed")
    )
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[_ExtractedNode(ref="pending", label="Pending", type="concept")]
            )
        ]
    )

    synchronize_documents(
        {
            "docs/processed.md": "# Processed\nText.",
            "docs/pending.md": "# Pending\nNeeds validation.",
        },
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
    )

    graph = knowledge_base_store.load()
    pending_chunk_id = f"{source_key('docs/pending.md')}-chunk-0000"
    assert {node.id for node in graph.nodes} == {
        "processed",
        f"{pending_chunk_id}::node-0001",
    }


def test_synchronize_changed_document_replaces_only_its_contributions_and_caches(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    graph_embedding_store = JsonGraphEmbeddingStore(tmp_path / "graph_embeddings")
    knowledge_base_embedding_store = JsonEmbeddingStore(
        tmp_path / "knowledge_base_embeddings.json"
    )
    chunk_embedding_store = JsonChunkEmbeddingStore(
        tmp_path / "chunk_embeddings.json"
    )
    taxonomy_store = JsonTaxonomySuggestionStore(
        tmp_path / "taxonomy_suggestions.json"
    )
    source_a = "docs/a.md"
    source_b = "docs/b.md"
    original_a = "# A\nShared has Old A."
    content_b = "# B\nShared has B."
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="old-a", label="Old A", type="feature"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="a-edge",
                        source_ref="shared",
                        target_ref="old-a",
                        label="has",
                    )
                ],
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="b", label="B", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="b-edge",
                        source_ref="shared",
                        target_ref="b",
                        label="has",
                    )
                ],
            ),
        ],
        decisions=[],
        vectors={},
    )
    synchronize_documents(
        {source_a: original_a, source_b: content_b},
        graph_store,
        chunk_store,
        initial,
        knowledge_base_store=knowledge_base_store,
        graph_embedding_store=graph_embedding_store,
        knowledge_base_embedding_store=knowledge_base_embedding_store,
        chunk_embedding_store=chunk_embedding_store,
        taxonomy_suggestion_store=taxonomy_store,
    )
    old_a_chunk_id = f"{source_key(source_a)}-chunk-0000"
    b_chunk_id = f"{source_key(source_b)}-chunk-0000"
    chunk_embedding_store.save(
        {
            old_a_chunk_id: ChunkEmbeddingRecord(
                chunk_id=old_a_chunk_id,
                embedding_model="model",
                embedding_input_hash="old-a",
                vector=[1.0],
            ),
            b_chunk_id: ChunkEmbeddingRecord(
                chunk_id=b_chunk_id,
                embedding_model="model",
                embedding_input_hash="b",
                vector=[1.0],
            ),
        }
    )
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    taxonomy_store.save(
        [
            TaxonomySuggestionRecord(
                suggested_type="old-a",
                normalized_suggested_type="old a",
                node_id="old-a",
                node_label="Old A",
                current_type="unclassified",
                source=source_a,
                chunk_id=old_a_chunk_id,
                created_at=timestamp,
            ),
            TaxonomySuggestionRecord(
                suggested_type="b",
                normalized_suggested_type="b",
                node_id="b",
                node_label="B",
                current_type="unclassified",
                source=source_b,
                chunk_id=b_chunk_id,
                created_at=timestamp,
            ),
        ]
    )
    original_graph = knowledge_base_store.load()
    shared_id = next(node.id for node in original_graph.nodes if node.label == "Shared")
    replacement_a = "# A\nShared has New A."
    changed = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(
                        ref="new-a",
                        label="New A",
                        type="capability",
                    ),
                ],
                edges=[
                    _ExtractedEdge(
                        id="new-a-edge",
                        source_ref="shared",
                        target_ref="new-a",
                        label="has",
                    )
                ],
            )
        ],
        decisions=[],
        vectors={},
    )

    result = synchronize_documents(
        {source_a: replacement_a, source_b: content_b},
        graph_store,
        chunk_store,
        changed,
        knowledge_base_store=knowledge_base_store,
        graph_embedding_store=graph_embedding_store,
        knowledge_base_embedding_store=knowledge_base_embedding_store,
        chunk_embedding_store=chunk_embedding_store,
        taxonomy_suggestion_store=taxonomy_store,
    )

    graph = knowledge_base_store.load()
    labels = {node.label for node in graph.nodes}
    shared = next(node for node in graph.nodes if node.label == "Shared")
    assert result.changed_sources == [source_a]
    assert result.unchanged_sources == [source_b]
    assert len(changed.responses) == 1
    assert shared.id == shared_id
    assert [item.source for item in shared.provenance] == [source_b, source_a]
    assert "Old A" not in labels
    assert {"Shared", "New A", "B"} <= labels
    assert graph_store.load(source_a).metadata.content_hash == document_content_hash(
        replacement_a
    )
    assert set(chunk_embedding_store.load()) == {b_chunk_id}
    assert {record.source for record in taxonomy_store.load()} == {source_b}


def test_changed_local_node_id_does_not_overwrite_surviving_shared_entity(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    source_a = "docs/a.md"
    source_b = "docs/b.md"
    content_b = "# B\nShared."
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                [_ExtractedNode(ref="shared", label="Shared", type="concept")]
            ),
            _extracted_graph(
                [_ExtractedNode(ref="shared", label="Shared", type="concept")]
            ),
        ],
        decisions=[],
        vectors={},
    )
    synchronize_documents(
        {source_a: "# A\nShared.", source_b: content_b},
        graph_store,
        chunk_store,
        initial,
        knowledge_base_store=knowledge_base_store,
    )
    shared_id = knowledge_base_store.load().nodes[0].id
    changed = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                [
                    _ExtractedNode(
                        ref="different",
                        label="Different",
                        type="capability",
                    )
                ]
            )
        ],
        decisions=[],
        vectors={},
    )

    synchronize_documents(
        {source_a: "# A\nDifferent.", source_b: content_b},
        graph_store,
        chunk_store,
        changed,
        knowledge_base_store=knowledge_base_store,
    )

    graph = knowledge_base_store.load()
    assert {node.label for node in graph.nodes} == {"Shared", "Different"}
    assert next(node.id for node in graph.nodes if node.label == "Shared") == shared_id
    assert len({node.id for node in graph.nodes}) == 2


def test_synchronize_deleted_document_removes_artifacts_but_keeps_shared_nodes(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    graph_embedding_store = JsonGraphEmbeddingStore(tmp_path / "graph_embeddings")
    knowledge_base_embedding_store = JsonEmbeddingStore(
        tmp_path / "knowledge_base_embeddings.json"
    )
    source_a = "docs/a.md"
    source_b = "docs/b.md"
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="a", label="A", type="feature"),
                ]
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="b", label="B", type="tool"),
                ]
            ),
        ],
        decisions=[],
        vectors={},
    )
    synchronize_documents(
        {source_a: "# A\nShared and A.", source_b: "# B\nShared and B."},
        graph_store,
        chunk_store,
        initial,
        knowledge_base_store=knowledge_base_store,
        graph_embedding_store=graph_embedding_store,
        knowledge_base_embedding_store=knowledge_base_embedding_store,
    )
    shared_id = next(
        node.id
        for node in knowledge_base_store.load().nodes
        if node.label == "Shared"
    )
    deleting = FakeSemanticLLM(responses=[], decisions=[], vectors={})

    result = synchronize_documents(
        {source_b: "# B\nShared and B."},
        graph_store,
        chunk_store,
        deleting,
        knowledge_base_store=knowledge_base_store,
        graph_embedding_store=graph_embedding_store,
        knowledge_base_embedding_store=knowledge_base_embedding_store,
    )

    graph = knowledge_base_store.load()
    shared = next(node for node in graph.nodes if node.label == "Shared")
    assert result.deleted_sources == [source_a]
    assert graph_store.exists(source_a) is False
    with pytest.raises(FileNotFoundError):
        chunk_store.load(source_a)
    assert graph_embedding_store.exists(source_a) is False
    assert shared.id == shared_id
    assert [item.source for item in shared.provenance] == [source_b]
    assert {node.label for node in graph.nodes} == {"Shared", "B"}
    assert set(knowledge_base_embedding_store.load()) == {
        node.id for node in graph.nodes
    }


def test_synchronize_rename_is_delete_plus_add(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    old_source = "docs/old.md"
    new_source = "docs/new.md"
    content = "# Document\nContent."
    synchronize_documents(
        {old_source: content},
        graph_store,
        chunk_store,
        FakeLLM(
            [
                _extracted_graph(
                    [_ExtractedNode(ref="doc", label="Doc", type="document")]
                )
            ]
        ),
        knowledge_base_store=knowledge_base_store,
    )

    result = synchronize_documents(
        {new_source: content},
        graph_store,
        chunk_store,
        FakeLLM(
            [
                _extracted_graph(
                    [_ExtractedNode(ref="doc", label="Doc", type="document")]
                )
            ]
        ),
        knowledge_base_store=knowledge_base_store,
    )

    assert result.added_sources == [new_source]
    assert result.deleted_sources == [old_source]
    assert graph_store.exists(old_source) is False
    assert graph_store.exists(new_source) is True


def test_failed_changed_document_extraction_preserves_active_data(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    source = "docs/a.md"
    original = "# A\nOriginal."
    synchronize_documents(
        {source: original},
        graph_store,
        chunk_store,
        FakeLLM(
            [_extracted_graph([_ExtractedNode(ref="a", label="A", type="concept")])]
        ),
        knowledge_base_store=knowledge_base_store,
    )
    original_graph = graph_store.load(source)
    original_knowledge_base = knowledge_base_store.load()

    with pytest.raises(RuntimeError, match="extraction failed"):
        synchronize_documents(
            {source: "# A\nReplacement."},
            graph_store,
            chunk_store,
            FailingLLM(),
            knowledge_base_store=knowledge_base_store,
        )

    assert graph_store.load(source) == original_graph
    assert knowledge_base_store.load() == original_knowledge_base
    assert chunk_store.load(source)[0].text == original


def test_search_knowledge_base_uses_cached_graph_when_available(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    chunk = _chunk("docs/cached.md", "# Ordinary\nPlain text.", "Ordinary")
    chunk_store.save("docs/cached.md", [chunk])
    graph_store.save(_graph("docs/cached.md", chunk, "stored", "Stored"))
    knowledge_base_store.save(
        KnowledgeGraph(
            nodes=[
                Node(
                    id="cached",
                    label="cacheonly",
                    type="Concept",
                    chunk_ids=[chunk.id],
                )
            ],
            edges=[],
        )
    )

    result = search_knowledge_base(
        "cacheonly",
        graph_store,
        chunk_store,
        knowledge_base_store=knowledge_base_store,
    )

    assert [hit.chunk.id for hit in result.chunks] == [chunk.id]
    assert [node.id for node in result.chunks[0].linked_nodes] == ["cached"]


def test_search_knowledge_base_uses_chunk_embeddings_when_available(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    chunk_embedding_store = JsonChunkEmbeddingStore(
        tmp_path / "chunk_embeddings.json"
    )
    source = "docs/deploy.md"
    chunk = _chunk(
        source,
        "# Deploy\nSetup stalls after authentication.",
        "Deploy",
    )
    chunk_store.save(source, [chunk])
    graph_store.save(
        KnowledgeGraph(
            nodes=[],
            edges=[],
            metadata=GraphMetadata(
                source=source,
                content_hash=document_content_hash(chunk.text),
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        )
    )
    embedding = FakeEmbeddingClient(
        {
            "install hangs": [1.0, 0.0],
            "Setup stalls": [1.0, 0.0],
        }
    )

    result = search_knowledge_base(
        "install hangs",
        graph_store,
        chunk_store,
        embedding_client=embedding,
        chunk_embedding_store=chunk_embedding_store,
    )

    assert [hit.chunk.id for hit in result.chunks] == [chunk.id]
    assert chunk_embedding_store.exists() is True


def test_search_knowledge_base_raises_when_saved_graph_has_no_chunks(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    chunk = _chunk("docs/missing.md", "# Missing\nValidation.", "Missing")
    graph_store.save(_graph("docs/missing.md", chunk, "missing", "Missing"))

    with pytest.raises(FileNotFoundError):
        search_knowledge_base("validation", graph_store, chunk_store)


def test_load_search_context_loads_cached_graph_and_all_chunks(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    pydantic_chunk = _chunk(
        "docs/pydantic.md", "# Pydantic\nValidation.", "Pydantic"
    )
    fastapi_chunk = _chunk(
        "docs/fastapi.md", "# FastAPI\nValidation.", "FastAPI"
    )
    chunk_store.save("docs/pydantic.md", [pydantic_chunk])
    chunk_store.save("docs/fastapi.md", [fastapi_chunk])
    knowledge_base = KnowledgeGraph(
        nodes=[
            Node(
                id="kb-node",
                label="KB",
                type="Concept",
                chunk_ids=[pydantic_chunk.id],
            )
        ],
        edges=[],
    )
    knowledge_base_store.save(knowledge_base)

    context = load_search_context(
        graph_store, chunk_store, knowledge_base_store=knowledge_base_store
    )

    assert context.graph == knowledge_base
    assert {chunk.id for chunk in context.chunks} == {
        pydantic_chunk.id,
        fastapi_chunk.id,
    }


def test_load_search_context_falls_back_to_document_graphs_without_kb_store(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    chunk = _chunk("docs/guide.md", "# Guide\nContent.", "Guide")
    chunk_store.save("docs/guide.md", [chunk])
    graph_store.save(_graph("docs/guide.md", chunk, "guide", "Guide"))

    context = load_search_context(graph_store, chunk_store)

    assert [chunk.id for chunk in context.chunks] == [chunk.id]
    assert any(node.label == "Guide" for node in context.graph.nodes)


def test_load_search_context_raises_when_graph_has_no_chunks(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    chunk = _chunk("docs/missing.md", "# Missing\nValidation.", "Missing")
    graph_store.save(_graph("docs/missing.md", chunk, "missing", "Missing"))

    with pytest.raises(FileNotFoundError):
        load_search_context(graph_store, chunk_store)
