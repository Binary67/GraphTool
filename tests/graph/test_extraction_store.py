import json

import pytest
from pydantic import ValidationError

from graphtool.graph.extraction_store import (
    ExtractedKnowledgeGraph,
    ExtractedNode,
    JsonChunkExtractionStore,
)


def _graph(label: str) -> ExtractedKnowledgeGraph:
    return ExtractedKnowledgeGraph(
        nodes=[ExtractedNode(ref=label.casefold(), label=label, type="concept")],
        edges=[],
    )


def test_chunk_extraction_store_round_trips_and_replaces_records(tmp_path):
    store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    source = "docs/guide.md"

    store.replace(source, {"first": _graph("First"), "stale": _graph("Stale")})
    store.replace(source, {"first": _graph("First"), "second": _graph("Second")})

    records = store.load(source)
    assert list(records) == ["first", "second"]
    assert records["first"].nodes[0].label == "First"
    assert records["second"].nodes[0].label == "Second"


def test_chunk_extraction_store_deletes_source_manifest(tmp_path):
    store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    source = "docs/guide.md"
    store.replace(source, {"first": _graph("First")})

    store.delete(source)

    assert store.load(source) == {}


def test_chunk_extraction_store_rejects_malformed_records(tmp_path):
    directory = tmp_path / "chunk_extractions"
    store = JsonChunkExtractionStore(directory)
    store.replace("docs/guide.md", {"first": _graph("First")})
    path = next(directory.glob("*.json"))
    data = json.loads(path.read_text())
    data["records"]["first"]["nodes"][0]["unexpected"] = True
    path.write_text(json.dumps(data))

    with pytest.raises(ValidationError):
        store.load("docs/guide.md")
