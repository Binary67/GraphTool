from datetime import datetime, timezone
from typing import TypeVar

import pytest

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.corpus import (
    filter_unprocessed_sources,
    ingest_unprocessed_documents,
    load_markdown_documents,
    search_knowledge_base,
)
from graphtool.graph.json_store import JsonGraphStore
from graphtool.graph.types import GraphMetadata, KnowledgeGraph, Node
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


def test_search_knowledge_base_raises_when_saved_graph_has_no_chunks(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    chunk_store = JsonChunkStore(tmp_path / "chunks")
    chunk = _chunk("docs/missing.md", "# Missing\nValidation.", "Missing")
    graph_store.save(_graph("docs/missing.md", chunk, "missing", "Missing"))

    with pytest.raises(FileNotFoundError):
        search_knowledge_base("validation", graph_store, chunk_store)
