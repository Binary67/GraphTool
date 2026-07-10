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
        Chunk(
            id="doc-chunk-0002",
            source="doc.md",
            index=2,
            text="## Ending\nFinal text.",
            heading_path=["Intro", "Ending"],
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


@pytest.mark.parametrize(
    ("chunk_id", "expected_positions"),
    [
        ("doc-chunk-0000", (None, 0, 1)),
        ("doc-chunk-0001", (0, 1, 2)),
        ("doc-chunk-0002", (1, 2, None)),
    ],
)
def test_load_neighborhood_handles_first_middle_and_last_chunks(
    tmp_path,
    chunk_id,
    expected_positions,
):
    store = JsonChunkStore(tmp_path)
    store.save("doc.md", _chunks())

    neighborhood = store.load_neighborhood("doc.md", chunk_id)

    assert tuple(
        chunk.index if chunk is not None else None
        for chunk in neighborhood
    ) == expected_positions


@pytest.mark.parametrize(
    ("source", "chunk_id"),
    [
        ("doc.md", "other-chunk-0000"),
        ("other.md", "doc-chunk-0001"),
    ],
)
def test_load_neighborhood_rejects_invalid_source_chunk_combinations(
    tmp_path,
    source,
    chunk_id,
):
    store = JsonChunkStore(tmp_path)
    store.save("doc.md", _chunks())
    store.save("other.md", [])

    with pytest.raises(ValueError, match="was not found in source"):
        store.load_neighborhood(source, chunk_id)


def test_save_uses_source_path_in_filename(tmp_path):
    store = JsonChunkStore(tmp_path)

    store.save("docs/api/guide.md", [])
    store.save("docs/user/guide.md", [])

    assert (tmp_path / f"{source_key('docs/api/guide.md')}.json").exists()
    assert (tmp_path / f"{source_key('docs/user/guide.md')}.json").exists()
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_delete_removes_saved_chunks(tmp_path):
    store = JsonChunkStore(tmp_path)
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Text",
        )
    ]
    store.save("doc.md", chunks)

    store.delete("doc.md")

    with pytest.raises(FileNotFoundError):
        store.load("doc.md")
