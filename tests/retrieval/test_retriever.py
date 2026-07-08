from datetime import datetime, timezone

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.retrieval import retrieve_context


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
    assert "pydantic" in top_chunk.linked_node_ids
    assert "edge-0001" in top_chunk.linked_edge_ids


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
    assert "Relevant nodes:\n- None" in result.context_text
    assert "Relevant relationships:\n- None" in result.context_text
    assert "Evidence:\n- None" in result.context_text
