from datetime import datetime, timezone

from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.provenance import (
    filter_knowledge_graph_by_source,
    remove_source_from_knowledge_graph,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node


def _graph(
    source: str,
    content_hash: str,
    *,
    label: str,
    node_type: str,
    chunk_id: str,
    edge_id: str,
) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="shared",
                label=label,
                type=node_type,
                properties={"source": source},
                chunk_ids=[chunk_id],
            )
        ],
        edges=[
            Edge(
                id=edge_id,
                source="shared",
                target="shared",
                label="supports",
                properties={"source": source},
                chunk_ids=[chunk_id],
            )
        ],
        metadata=GraphMetadata(
            source=source,
            content_hash=content_hash,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_combined_graph_records_and_materializes_document_provenance():
    first = _graph(
        "docs/a.md",
        "hash-a",
        label="Shared",
        node_type="unclassified",
        chunk_id="a-chunk",
        edge_id="a-edge",
    )
    second = _graph(
        "docs/b.md",
        "hash-b",
        label="Shared Entity",
        node_type="concept",
        chunk_id="b-chunk",
        edge_id="b-edge",
    )

    graph = combine_knowledge_graphs([first, second])

    assert graph.nodes[0].id == "shared"
    assert graph.nodes[0].type == "concept"
    assert graph.nodes[0].aliases == ["Shared Entity"]
    assert graph.nodes[0].chunk_ids == ["a-chunk", "b-chunk"]
    assert [item.source for item in graph.nodes[0].provenance] == [
        "docs/a.md",
        "docs/b.md",
    ]
    assert [item.source for item in graph.edges[0].provenance] == [
        "docs/a.md",
        "docs/b.md",
    ]
    assert graph.edges[0].provenance[0].source_node_id == "shared"
    assert graph.edges[0].provenance[0].target_node_id == "shared"


def test_removing_source_recalculates_shared_elements_and_preserves_ids():
    graph = combine_knowledge_graphs(
        [
            _graph(
                "docs/a.md",
                "hash-a",
                label="First Label",
                node_type="unclassified",
                chunk_id="a-chunk",
                edge_id="a-edge",
            ),
            _graph(
                "docs/b.md",
                "hash-b",
                label="Remaining Label",
                node_type="concept",
                chunk_id="b-chunk",
                edge_id="b-edge",
            ),
        ]
    )

    remaining = remove_source_from_knowledge_graph(graph, "docs/a.md")

    assert remaining.nodes[0].id == "shared"
    assert remaining.nodes[0].label == "Remaining Label"
    assert remaining.nodes[0].type == "concept"
    assert remaining.nodes[0].properties == {"source": "docs/b.md"}
    assert remaining.nodes[0].chunk_ids == ["b-chunk"]
    assert remaining.edges[0].id == graph.edges[0].id
    assert remaining.edges[0].properties == {"source": "docs/b.md"}
    assert remaining.edges[0].chunk_ids == ["b-chunk"]
    assert remove_source_from_knowledge_graph(remaining, "docs/b.md") == (
        KnowledgeGraph(nodes=[], edges=[])
    )


def test_filter_knowledge_graph_by_source_returns_valid_source_view():
    graph = combine_knowledge_graphs(
        [
            _graph(
                "docs/a.md",
                "hash-a",
                label="First Label",
                node_type="concept",
                chunk_id="a-chunk",
                edge_id="a-edge",
            ),
            _graph(
                "docs/b.md",
                "hash-b",
                label="Second Label",
                node_type="concept",
                chunk_id="b-chunk",
                edge_id="b-edge",
            ),
        ]
    )

    filtered = filter_knowledge_graph_by_source(graph, "docs/b.md")

    assert filtered.nodes[0].id == "shared"
    assert filtered.nodes[0].label == "Second Label"
    assert filtered.nodes[0].chunk_ids == ["b-chunk"]
    assert [item.source for item in filtered.nodes[0].provenance] == ["docs/b.md"]
    assert filtered.edges[0].source == "shared"
    assert filtered.edges[0].target == "shared"
    assert filtered.edges[0].chunk_ids == ["b-chunk"]
