import json
from datetime import datetime, timezone

from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.source import source_key


def _sample_graph(source: str = "doc.md") -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[Node(id="a", label="A", type="Concept")],
        edges=[Edge(id="e1", source="a", target="a", label="relates_to")],
        metadata=GraphMetadata(
            source=source,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_save_creates_json_file(tmp_path):
    store = JsonGraphStore(tmp_path)

    store.save(_sample_graph("doc.md"))

    expected = tmp_path / f"{source_key('doc.md')}.json"
    assert expected.exists()


def test_save_writes_valid_json_with_nodes_and_edges(tmp_path):
    store = JsonGraphStore(tmp_path)

    store.save(_sample_graph("doc.md"))

    data = json.loads((tmp_path / f"{source_key('doc.md')}.json").read_text())
    assert data["nodes"][0]["id"] == "a"
    assert data["edges"][0]["source"] == "a"
    assert data["metadata"]["source"] == "doc.md"


def test_save_writes_node_and_edge_chunk_ids(tmp_path):
    store = JsonGraphStore(tmp_path)
    graph = KnowledgeGraph(
        nodes=[Node(id="a", label="A", type="Concept", chunk_ids=["doc-chunk-0000"])],
        edges=[
            Edge(
                id="e1",
                source="a",
                target="a",
                label="relates_to",
                chunk_ids=["doc-chunk-0000"],
            )
        ],
        metadata=GraphMetadata(
            source="doc.md",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )

    store.save(graph)

    data = json.loads((tmp_path / f"{source_key('doc.md')}.json").read_text())
    assert data["nodes"][0]["chunk_ids"] == ["doc-chunk-0000"]
    assert data["edges"][0]["chunk_ids"] == ["doc-chunk-0000"]


def test_load_roundtrips_saved_graph(tmp_path):
    store = JsonGraphStore(tmp_path)
    original = _sample_graph("doc.md")

    store.save(original)
    loaded = store.load("doc.md")

    assert loaded == original


def test_load_raises_for_missing_file(tmp_path):
    import pytest

    store = JsonGraphStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("missing.md")


def test_save_raises_clear_error_without_metadata(tmp_path):
    import pytest

    store = JsonGraphStore(tmp_path)
    graph = KnowledgeGraph(nodes=[], edges=[])

    with pytest.raises(ValueError, match="metadata.source"):
        store.save(graph)


def test_save_creates_directory_if_missing(tmp_path):
    store = JsonGraphStore(tmp_path / "nested" / "graphs")

    store.save(_sample_graph("doc.md"))

    assert (
        tmp_path / "nested" / "graphs" / f"{source_key('doc.md')}.json"
    ).exists()


def test_exists_returns_true_only_for_saved_source(tmp_path):
    store = JsonGraphStore(tmp_path)

    store.save(_sample_graph("doc.md"))

    assert store.exists("doc.md") is True
    assert store.exists("missing.md") is False


def test_load_all_returns_saved_graphs_in_filename_order(tmp_path):
    store = JsonGraphStore(tmp_path)
    first = _sample_graph("docs/api/guide.md")
    second = _sample_graph("docs/user/guide.md")

    store.save(first)
    store.save(second)

    loaded = store.load_all()

    assert sorted(graph.metadata.source for graph in loaded if graph.metadata) == [
        "docs/api/guide.md",
        "docs/user/guide.md",
    ]


def test_save_uses_source_path_in_filename(tmp_path):
    store = JsonGraphStore(tmp_path)

    store.save(_sample_graph("docs/api/guide.md"))
    store.save(_sample_graph("docs/user/guide.md"))

    assert (tmp_path / f"{source_key('docs/api/guide.md')}.json").exists()
    assert (tmp_path / f"{source_key('docs/user/guide.md')}.json").exists()
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_knowledge_base_store_roundtrips_metadata_less_graph(tmp_path):
    store = JsonKnowledgeBaseStore(tmp_path / "nested" / "knowledge_base.json")
    graph = KnowledgeGraph(
        nodes=[Node(id="a", label="A", type="Concept")],
        edges=[Edge(id="edge-0001", source="a", target="a", label="relates_to")],
    )

    store.save(graph)
    loaded = store.load()

    assert store.exists() is True
    assert loaded == graph
    assert (tmp_path / "nested" / "knowledge_base.json").exists()
