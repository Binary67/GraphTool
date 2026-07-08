import pytest
from pydantic import ValidationError

from graphtool.graph.types import Edge, KnowledgeGraph, Node


def test_valid_graph_passes_validation():
    graph = KnowledgeGraph(
        nodes=[
            Node(id="python", label="Python", type="Language"),
            Node(id="pydantic", label="Pydantic", type="Library"),
        ],
        edges=[
            Edge(
                id="e1",
                source="pydantic",
                target="python",
                label="built_for",
            )
        ],
    )

    assert graph.edges[0].source == "pydantic"


def test_self_loop_passes_when_node_exists():
    graph = KnowledgeGraph(
        nodes=[Node(id="python", label="Python", type="Language")],
        edges=[Edge(id="e1", source="python", target="python", label="self")],
    )

    assert graph.edges[0].target == "python"


def test_edge_with_missing_source_fails_validation():
    with pytest.raises(ValidationError, match="missing source node"):
        KnowledgeGraph(
            nodes=[Node(id="python", label="Python", type="Language")],
            edges=[Edge(id="e1", source="java", target="python", label="mentions")],
        )


def test_edge_with_missing_target_fails_validation():
    with pytest.raises(ValidationError, match="missing target node"):
        KnowledgeGraph(
            nodes=[Node(id="python", label="Python", type="Language")],
            edges=[Edge(id="e1", source="python", target="java", label="mentions")],
        )


def test_duplicate_node_ids_fail_validation():
    with pytest.raises(ValidationError, match="node ids must be unique"):
        KnowledgeGraph(
            nodes=[
                Node(id="python", label="Python", type="Language"),
                Node(id="python", label="Python 3", type="Version"),
            ],
            edges=[],
        )


def test_duplicate_edge_ids_fail_validation():
    with pytest.raises(ValidationError, match="edge ids must be unique"):
        KnowledgeGraph(
            nodes=[
                Node(id="python", label="Python", type="Language"),
                Node(id="pydantic", label="Pydantic", type="Library"),
            ],
            edges=[
                Edge(id="e1", source="python", target="pydantic", label="uses"),
                Edge(id="e1", source="pydantic", target="python", label="supports"),
            ],
        )


def test_empty_graph_passes_validation():
    graph = KnowledgeGraph(nodes=[], edges=[])

    assert graph.nodes == []
    assert graph.edges == []


def test_node_aliases_and_chunk_ids_default_empty():
    node = Node(id="python", label="Python", type="Language")
    edge = Edge(id="e1", source="python", target="python", label="self")

    assert node.aliases == []
    assert node.chunk_ids == []
    assert edge.chunk_ids == []
