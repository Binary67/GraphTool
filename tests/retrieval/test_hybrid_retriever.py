from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.retrieval import retrieve_context, retrieve_hybrid_context


def _corpus() -> tuple[KnowledgeGraph, list[Chunk]]:
    chunks = [
        Chunk(
            id="alpha-beta",
            source="doc.md",
            index=0,
            text="Alpha uses Beta.",
        ),
        Chunk(
            id="beta-gamma",
            source="doc.md",
            index=1,
            text="Beta depends on Gamma.",
        ),
        Chunk(
            id="operations",
            source="operations.md",
            index=0,
            text="Retry throttled deployment operations.",
        ),
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="alpha",
                label="Alpha",
                type="System",
                chunk_ids=["alpha-beta"],
            ),
            Node(
                id="beta",
                label="Beta",
                type="Component",
                chunk_ids=["alpha-beta", "beta-gamma"],
            ),
            Node(
                id="gamma",
                label="Gamma",
                type="Service",
                chunk_ids=["beta-gamma"],
            ),
        ],
        edges=[
            Edge(
                id="alpha-beta-edge",
                source="alpha",
                target="beta",
                label="uses",
                chunk_ids=["alpha-beta"],
            ),
            Edge(
                id="beta-gamma-edge",
                source="beta",
                target="gamma",
                label="depends on",
                chunk_ids=["beta-gamma"],
            ),
        ],
    )
    return graph, chunks


def test_hybrid_search_fuses_unique_chunks_and_preserves_graph_paths():
    graph, chunks = _corpus()

    result = retrieve_hybrid_context(
        "How is Alpha related to Gamma?",
        graph,
        chunks,
        max_hops=2,
        top_paths=1,
        top_chunks=3,
    )

    chunk_ids = [hit.chunk.id for hit in result.chunks]
    assert chunk_ids == ["alpha-beta", "beta-gamma"]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(result.graph_paths) == 1
    assert [edge.id for edge in result.graph_paths[0].edges] == [
        "alpha-beta-edge",
        "beta-gamma-edge",
    ]
    assert "Graph paths:" in result.context_text


def test_hybrid_search_falls_back_to_direct_chunks_without_graph_match():
    graph, chunks = _corpus()

    direct = retrieve_context("throttled deployment", graph, chunks)
    hybrid = retrieve_hybrid_context("throttled deployment", graph, chunks)

    assert [hit.chunk.id for hit in hybrid.chunks] == [
        hit.chunk.id for hit in direct.chunks
    ]
    assert hybrid.graph_paths == []
    assert hybrid.sources == ["operations.md"]
