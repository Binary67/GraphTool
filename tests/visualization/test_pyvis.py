from datetime import datetime, timezone

from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.visualization import export_graph_html


def _sample_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="python",
                label="Python",
                type="Language",
                properties={"version": "3.13"},
                chunk_ids=["doc-chunk-0000"],
            ),
            Node(
                id="pydantic",
                label="Pydantic",
                type="Library",
                properties={"purpose": "validation"},
                chunk_ids=["doc-chunk-0001"],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="pydantic",
                target="python",
                label="supports",
                properties={"confidence": "high"},
                chunk_ids=["doc-chunk-0000", "doc-chunk-0001"],
            )
        ],
        metadata=GraphMetadata(
            source="doc.md",
            content_hash="hash",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_export_graph_html_writes_interactive_html(tmp_path):
    output_path = tmp_path / "nested" / "graph.html"

    returned_path = export_graph_html(_sample_graph(), output_path)

    assert returned_path == output_path.resolve()
    assert output_path.exists()
    html = output_path.read_text()
    assert "Python" in html
    assert "Pydantic" in html
    assert "supports" in html
    assert "Language" in html
    assert "edge-0001" in html
    assert "version" in html
    assert "confidence" in html
    assert "doc-chunk-0000" in html
    assert "doc-chunk-0001" in html


def test_export_graph_html_allows_empty_graph(tmp_path):
    output_path = tmp_path / "empty.html"
    graph = KnowledgeGraph(nodes=[], edges=[])

    returned_path = export_graph_html(graph, output_path)

    assert returned_path == output_path.resolve()
    assert output_path.exists()
    assert "<html>" in output_path.read_text()


def test_export_graph_html_is_importable_from_visualization_package():
    assert callable(export_graph_html)
    assert export_graph_html.__name__ == "export_graph_html"
