import pytest

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.types import Chunk
from graphtool.source import source_key


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="# Intro\nText.",
            heading_path=["Intro"],
        ),
        Chunk(
            id="doc-chunk-0001",
            source="doc.md",
            index=1,
            text="## Details\nMore text.",
            heading_path=["Intro", "Details"],
        ),
    ]


def test_save_creates_json_file(tmp_path):
    store = JsonChunkStore(tmp_path)

    store.save("doc.md", _chunks())

    assert (tmp_path / f"{source_key('doc.md')}.json").exists()


def test_load_roundtrips_saved_chunks(tmp_path):
    store = JsonChunkStore(tmp_path)
    chunks = _chunks()

    store.save("doc.md", chunks)
    loaded = store.load("doc.md")

    assert loaded == chunks


def test_load_raises_for_missing_file(tmp_path):
    store = JsonChunkStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("missing.md")


def test_load_by_ids_returns_requested_order_and_filters_missing_ids(tmp_path):
    store = JsonChunkStore(tmp_path)
    chunks = _chunks()
    store.save("doc.md", chunks)

    loaded = store.load_by_ids(
        "doc.md",
        ["doc-chunk-0001", "missing-chunk", "doc-chunk-0000"],
    )

    assert loaded == [chunks[1], chunks[0]]


def test_save_uses_source_path_in_filename(tmp_path):
    store = JsonChunkStore(tmp_path)

    store.save("docs/api/guide.md", [])
    store.save("docs/user/guide.md", [])

    assert (tmp_path / f"{source_key('docs/api/guide.md')}.json").exists()
    assert (tmp_path / f"{source_key('docs/user/guide.md')}.json").exists()
    assert len(list(tmp_path.glob("*.json"))) == 2
