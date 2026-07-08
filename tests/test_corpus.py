from datetime import datetime, timezone
from typing import TypeVar

import pytest

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.corpus import (
    filter_unprocessed_sources,
    ingest_unprocessed_documents,
    load_markdown_documents,
    rebuild_knowledge_base,
    search_knowledge_base,
)
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage
from graphtool.source import source_key

T = TypeVar("T")


class FakeLLM:
    def __init__(self, responses: list[KnowledgeGraph]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return self.responses[len(self.calls) - 1]


def _chunk(source: str, text: str, heading: str) -> Chunk:
    return Chunk(
        id=f"{source_key(source)}-chunk-0000",
        source=source,
        index=0,
        text=text,
        heading_path=[heading],
    )


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
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def _relationship_graph(source: str, chunk_id: str) -> KnowledgeGraph:
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


def test_filter_unprocessed_sources_skips_saved_graphs(tmp_path):
    graph_store = JsonGraphStore(tmp_path)
    chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(_graph("docs/processed.md", chunk, "processed", "Processed"))

    unprocessed = filter_unprocessed_sources(
        ["docs/processed.md", "docs/pending.md"],
        graph_store,
    )

    assert unprocessed == ["docs/pending.md"]


def test_ingest_unprocessed_documents_skips_processed_sources(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    processed_chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(
        _graph("docs/processed.md", processed_chunk, "processed", "Processed")
    )
    fake = FakeLLM(
        [
            KnowledgeGraph(
                nodes=[Node(id="pending", label="Pending", type="Concept")],
                edges=[],
            )
        ]
    )

    graphs = ingest_unprocessed_documents(
        {
            "docs/processed.md": "# Processed\nText.",
            "docs/pending.md": "# Pending\nNeeds validation.",
        },
        graph_store,
        chunk_store,
        fake,
    )

    assert len(graphs) == 1
    assert graphs[0].metadata is not None
    assert graphs[0].metadata.source == "docs/pending.md"
    assert len(fake.calls) == 1
    assert graph_store.exists("docs/pending.md") is True
    assert chunk_store.load("docs/pending.md")


def test_ingest_unprocessed_documents_updates_cached_knowledge_base_with_exact_dedupe(
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
            KnowledgeGraph(
                nodes=[
                    Node(id="graphtool", label="GraphTool", type="Project"),
                    Node(id="azure-openai", label="Azure OpenAI", type="Service"),
                ],
                edges=[
                    Edge(
                        id="llm-edge",
                        source="graphtool",
                        target="azure-openai",
                        label="uses",
                    )
                ],
            )
        ]
    )
    new_source = "docs/new.md"
    new_chunk_id = f"{source_key(new_source)}-chunk-0000"

    ingest_unprocessed_documents(
        {new_source: "# GraphTool\nUses Azure OpenAI."},
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
    )

    graph = knowledge_base_store.load()
    assert len(graph.nodes) == 2
    assert graph.nodes[0].chunk_ids == ["existing-chunk-0000", new_chunk_id]
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["existing-chunk-0000", new_chunk_id]


def test_ingest_unprocessed_documents_rebuilds_missing_knowledge_base_cache(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    processed_chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(
        _graph("docs/processed.md", processed_chunk, "processed", "Processed")
    )
    fake = FakeLLM(
        [
            KnowledgeGraph(
                nodes=[Node(id="pending", label="Pending", type="Concept")],
                edges=[],
            )
        ]
    )

    ingest_unprocessed_documents(
        {"docs/pending.md": "# Pending\nNeeds validation."},
        graph_store,
        chunk_store,
        fake,
        knowledge_base_store=knowledge_base_store,
    )

    graph = knowledge_base_store.load()
    assert {node.id for node in graph.nodes} == {"processed", "pending"}


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

    assert [hit.node.id for hit in result.node_hits] == ["cached"]


def test_search_knowledge_base_raises_when_saved_graph_has_no_chunks(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    chunk = _chunk("docs/missing.md", "# Missing\nValidation.", "Missing")
    graph_store.save(_graph("docs/missing.md", chunk, "missing", "Missing"))

    with pytest.raises(FileNotFoundError):
        search_knowledge_base("validation", graph_store, chunk_store)
