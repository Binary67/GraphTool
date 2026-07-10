import json

from graphtool.agents.answer_questions.tools import (
    make_get_chunk_neighborhood_tool,
    make_retrieve_knowledge_context_tool,
)
from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.corpus import SearchContext
from graphtool.graph.types import KnowledgeGraph
from graphtool.retrieval.types import ChunkHit, RetrievalResult


def test_retrieve_knowledge_context_tool_returns_context_and_sources(monkeypatch):
    retrieve_calls = []

    chunk = Chunk(
        id="guide-chunk-0002",
        source="docs/guide.md",
        index=2,
        text="Relevant context",
        heading_path=["Guide", "Agents"],
    )
    graph = KnowledgeGraph(nodes=[], edges=[])

    def fake_load_search_context(graph_store, chunk_store, *, knowledge_base_store=None):
        return SearchContext(graph=graph, chunks=[chunk])

    def fake_retrieve_context(query, graph, chunks, **kwargs):
        retrieve_calls.append((query, graph, chunks, kwargs))
        return RetrievalResult(
            query=query,
            sources=["docs/guide.md"],
            chunks=[
                ChunkHit(
                    chunk=chunk,
                    score=1.0,
                    linked_nodes=[],
                    linked_relationships=[],
                )
            ],
            context_text="Relevant context",
        )

    monkeypatch.setattr(
        "graphtool.agents.answer_questions.tools.corpus.load_search_context",
        fake_load_search_context,
    )
    monkeypatch.setattr(
        "graphtool.agents.answer_questions.tools.retrieve_context",
        fake_retrieve_context,
    )
    graph_store = object()
    chunk_store = object()
    knowledge_base_store = object()
    embedding_client = object()
    chunk_embedding_store = object()
    allowed_chunks: set[tuple[str, str]] = set()
    tool = make_retrieve_knowledge_context_tool(
        graph_store,
        chunk_store,
        knowledge_base_store=knowledge_base_store,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
        allowed_chunks=allowed_chunks,
    )

    output = tool.invoke({"query": "What does GraphTool use?"})

    assert json.loads(output) == {
        "type": "search",
        "query": "What does GraphTool use?",
        "sources": ["docs/guide.md"],
        "chunk_references": [
            {
                "chunk_id": "guide-chunk-0002",
                "source": "docs/guide.md",
                "index": 2,
                "heading_path": ["Guide", "Agents"],
            }
        ],
        "context_text": "Relevant context",
    }
    assert allowed_chunks == {("docs/guide.md", "guide-chunk-0002")}
    assert retrieve_calls == [
        (
            "What does GraphTool use?",
            graph,
            [chunk],
            {
                "top_chunks": 5,
                "embedding_client": embedding_client,
                "chunk_embedding_store": chunk_embedding_store,
            },
        )
    ]


def test_get_chunk_neighborhood_tool_returns_typed_json(tmp_path):
    store = JsonChunkStore(tmp_path)
    chunks = [
        Chunk(
            id=f"guide-chunk-{index:04d}",
            source="docs/guide.md",
            index=index,
            text=f"Part {index}",
            heading_path=["Guide"],
        )
        for index in range(3)
    ]
    store.save("docs/guide.md", chunks)
    allowed_chunks = {("docs/guide.md", "guide-chunk-0001")}
    tool = make_get_chunk_neighborhood_tool(store, allowed_chunks=allowed_chunks)

    output = tool.invoke(
        {
            "source": "docs/guide.md",
            "chunk_id": "guide-chunk-0001",
        }
    )

    data = json.loads(output)
    assert data["type"] == "chunk_neighborhood"
    assert data["source"] == "docs/guide.md"
    assert data["chunk_id"] == "guide-chunk-0001"
    assert data["previous"]["chunk_id"] == "guide-chunk-0000"
    assert data["current"]["chunk_id"] == "guide-chunk-0001"
    assert data["next"]["chunk_id"] == "guide-chunk-0002"


def test_get_chunk_neighborhood_tool_rejects_unknown_chunk(tmp_path):
    store = JsonChunkStore(tmp_path)
    chunks = [
        Chunk(
            id=f"guide-chunk-{index:04d}",
            source="docs/guide.md",
            index=index,
            text=f"Part {index}",
            heading_path=["Guide"],
        )
        for index in range(3)
    ]
    store.save("docs/guide.md", chunks)
    tool = make_get_chunk_neighborhood_tool(store, allowed_chunks=set())

    output = tool.invoke(
        {
            "source": "docs/guide.md",
            "chunk_id": "guide-chunk-0001",
        }
    )

    data = json.loads(output)
    assert "error" in data
    assert "guide-chunk-0001" in data["error"]
