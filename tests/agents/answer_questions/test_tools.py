import json

from graphtool.agents.answer_questions.tools import make_retrieve_knowledge_context_tool
from graphtool.retrieval.types import RetrievalResult


def test_retrieve_knowledge_context_tool_returns_context_and_sources(monkeypatch):
    calls = []

    def fake_search_knowledge_base(query, graph_store, chunk_store, **kwargs):
        calls.append((query, graph_store, chunk_store, kwargs))
        return RetrievalResult(
            query=query,
            sources=["docs/guide.md"],
            chunks=[],
            context_text="Relevant context",
        )

    monkeypatch.setattr(
        "graphtool.agents.answer_questions.tools.corpus.search_knowledge_base",
        fake_search_knowledge_base,
    )
    graph_store = object()
    chunk_store = object()
    knowledge_base_store = object()
    embedding_client = object()
    chunk_embedding_store = object()
    tool = make_retrieve_knowledge_context_tool(
        graph_store,
        chunk_store,
        knowledge_base_store=knowledge_base_store,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
    )

    output = tool.invoke({"query": "What does GraphTool use?"})

    assert json.loads(output) == {
        "query": "What does GraphTool use?",
        "sources": ["docs/guide.md"],
        "context_text": "Relevant context",
    }
    assert calls == [
        (
            "What does GraphTool use?",
            graph_store,
            chunk_store,
            {
                "knowledge_base_store": knowledge_base_store,
                "embedding_client": embedding_client,
                "chunk_embedding_store": chunk_embedding_store,
                "top_chunks": 5,
            },
        )
    ]
