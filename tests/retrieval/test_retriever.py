from datetime import datetime, timezone

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.retrieval import JsonChunkEmbeddingStore, retrieve_context


class FakeEmbeddingClient:
    embedding_model = "fake-embedding-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.calls.extend(batch)
        self.batch_calls.append(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
        for marker, vector in self.vectors.items():
            if marker in text:
                return vector
        return [0.0, 1.0]


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="# Python\nPython is a programming language.",
            heading_path=["Python"],
        ),
        Chunk(
            id="doc-chunk-0001",
            source="doc.md",
            index=1,
            text="## Pydantic\nPydantic is a validation library built for Python.",
            heading_path=["Python", "Pydantic"],
        ),
        Chunk(
            id="doc-chunk-0002",
            source="doc.md",
            index=2,
            text="## FastAPI\nFastAPI uses Pydantic for request validation.",
            heading_path=["Python", "FastAPI"],
        ),
        Chunk(
            id="doc-chunk-0003",
            source="doc.md",
            index=3,
            text="## Django\nDjango includes an ORM.",
            heading_path=["Python", "Django"],
        ),
    ]


def _graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="python",
                label="Python",
                type="Language",
                chunk_ids=["doc-chunk-0000", "doc-chunk-0001"],
            ),
            Node(
                id="pydantic",
                label="Pydantic",
                type="Library",
                properties={"purpose": "data validation"},
                chunk_ids=["doc-chunk-0001", "missing-chunk"],
            ),
            Node(
                id="fastapi",
                label="FastAPI",
                type="Framework",
                chunk_ids=["doc-chunk-0002"],
            ),
            Node(
                id="django",
                label="Django",
                type="Framework",
                chunk_ids=["doc-chunk-0003"],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="pydantic",
                target="python",
                label="built_for",
                chunk_ids=["doc-chunk-0001"],
            ),
            Edge(
                id="edge-0002",
                source="fastapi",
                target="pydantic",
                label="uses",
                chunk_ids=["doc-chunk-0002"],
            ),
            Edge(
                id="edge-0003",
                source="django",
                target="python",
                label="includes",
                chunk_ids=["doc-chunk-0003"],
            ),
        ],
        metadata=GraphMetadata(
            source="doc.md",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_retrieve_context_retrieves_relevant_nodes_from_properties():
    result = retrieve_context(
        "data validation library",
        _graph(),
        _chunks(),
        top_nodes=2,
    )

    assert result.node_hits[0].node.id == "pydantic"
    assert result.node_hits[0].matched_text == "Pydantic"


def test_retrieve_context_retrieves_only_edges_connected_to_selected_nodes():
    result = retrieve_context(
        "data validation library",
        _graph(),
        _chunks(),
        top_nodes=1,
        top_edges=5,
    )

    edge_ids = {hit.edge.id for hit in result.relationship_hits}

    assert edge_ids
    assert edge_ids <= {"edge-0001", "edge-0002"}
    assert "edge-0003" not in edge_ids


def test_retrieve_context_boosts_chunks_linked_by_nodes_and_edges():
    result = retrieve_context(
        "data validation built for python",
        _graph(),
        _chunks(),
        top_nodes=2,
        top_edges=1,
        top_chunks=3,
    )

    top_chunk = result.chunks[0]

    assert top_chunk.chunk.id == "doc-chunk-0001"
    assert result.sources == ["doc.md"]
    assert "pydantic" in top_chunk.linked_node_ids
    assert "edge-0001" in top_chunk.linked_edge_ids


def test_retrieve_context_bm25_searches_chunks_without_node_matches():
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Deployment retries should reduce throttling errors.",
            heading_path=["Operations"],
        )
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="unrelated",
                label="Unrelated",
                type="Concept",
                chunk_ids=["doc-chunk-0000"],
            )
        ],
        edges=[],
    )

    result = retrieve_context("throttling errors", graph, chunks)

    assert result.node_hits == []
    assert [hit.chunk.id for hit in result.chunks] == ["doc-chunk-0000"]


def test_retrieve_context_uses_semantic_chunk_search(tmp_path):
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Setup stalls after authentication.",
            heading_path=["Deploy"],
        ),
        Chunk(
            id="doc-chunk-0001",
            source="doc.md",
            index=1,
            text="Billing exports finish normally.",
            heading_path=["Billing"],
        ),
    ]
    embedding = FakeEmbeddingClient(
        {
            "install hangs": [1.0, 0.0],
            "Setup stalls": [1.0, 0.0],
        }
    )
    store = JsonChunkEmbeddingStore(tmp_path / "chunk_embeddings.json")

    result = retrieve_context(
        "install hangs",
        KnowledgeGraph(nodes=[], edges=[]),
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
        top_chunks=1,
    )

    assert [hit.chunk.id for hit in result.chunks] == ["doc-chunk-0000"]
    assert store.exists() is True


def test_retrieve_context_reuses_and_refreshes_chunk_embedding_cache(tmp_path):
    store = JsonChunkEmbeddingStore(tmp_path / "chunk_embeddings.json")
    embedding = FakeEmbeddingClient(
        {
            "install hangs": [1.0, 0.0],
            "Setup stalls": [1.0, 0.0],
        }
    )
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Setup stalls after authentication.",
            heading_path=["Deploy"],
        )
    ]

    retrieve_context(
        "install hangs",
        KnowledgeGraph(nodes=[], edges=[]),
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
    )
    assert any("Setup stalls after authentication" in call for call in embedding.calls)

    embedding.calls.clear()
    retrieve_context(
        "install hangs",
        KnowledgeGraph(nodes=[], edges=[]),
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
    )
    assert embedding.calls == ["install hangs"]

    embedding.calls.clear()
    changed_chunks = [
        chunks[0].model_copy(
            update={"text": "Setup stalls after authentication recovery."}
        )
    ]
    retrieve_context(
        "install hangs",
        KnowledgeGraph(nodes=[], edges=[]),
        changed_chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
    )

    assert any("authentication recovery" in call for call in embedding.calls)


def test_retrieve_context_deduplicates_chunks_and_preserves_ranked_output():
    result = retrieve_context(
        "data validation built for python",
        _graph(),
        _chunks(),
        top_nodes=2,
        top_edges=1,
        top_chunks=3,
    )

    chunk_ids = [hit.chunk.id for hit in result.chunks]

    assert chunk_ids.count("doc-chunk-0001") == 1
    assert chunk_ids[0] == "doc-chunk-0001"


def test_retrieve_context_builds_agent_context_text():
    result = retrieve_context(
        "data validation built for python",
        _graph(),
        _chunks(),
        top_nodes=2,
        top_edges=1,
        top_chunks=1,
    )

    assert "Relevant nodes:" in result.context_text
    assert "Pydantic [Library]" in result.context_text
    assert "Pydantic --built_for--> Python" in result.context_text
    assert "[doc-chunk-0001 | doc.md | Python > Pydantic]" in result.context_text
    assert "Pydantic is a validation library built for Python." in result.context_text


def test_retrieve_context_returns_empty_hits_for_no_matches():
    result = retrieve_context("xylophone", _graph(), _chunks())

    assert result.node_hits == []
    assert result.relationship_hits == []
    assert result.chunks == []
    assert result.sources == []
    assert "Relevant nodes:\n- None" in result.context_text
    assert "Relevant relationships:\n- None" in result.context_text
    assert "Evidence:\n- None" in result.context_text
