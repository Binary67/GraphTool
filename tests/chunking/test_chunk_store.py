import pytest

from graphtool.chunking.store import SqliteChunkStore
from graphtool.chunking.types import Chunk
from graphtool.storage import open_database


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


def _store(tmp_path):
    return SqliteChunkStore(open_database(tmp_path / "test.db"))


def test_save_persists_chunks(tmp_path):
    store = _store(tmp_path)

    store.save("doc.md", _chunks())

    assert [chunk.id for chunk in store.load("doc.md")] == [
        "doc-chunk-0000",
        "doc-chunk-0001",
        "doc-chunk-0002",
    ]


def test_load_roundtrips_saved_chunks(tmp_path):
    store = _store(tmp_path)
    chunks = _chunks()

    store.save("doc.md", chunks)
    loaded = store.load("doc.md")

    assert loaded == chunks


def test_load_returns_empty_for_unsaved_source(tmp_path):
    store = _store(tmp_path)

    assert store.load("missing.md") == []


def test_load_by_ids_returns_requested_order_and_filters_missing_ids(tmp_path):
    store = _store(tmp_path)
    chunks = _chunks()
    store.save("doc.md", chunks)

    loaded = store.load_by_ids(
        "doc.md",
        ["doc-chunk-0001", "missing-chunk", "doc-chunk-0000"],
    )

    assert loaded == [chunks[1], chunks[0]]


def test_load_by_ids_returns_empty_for_empty_request(tmp_path):
    store = _store(tmp_path)
    store.save("doc.md", _chunks())

    assert store.load_by_ids("doc.md", []) == []


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
    store = _store(tmp_path)
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
    store = _store(tmp_path)
    store.save("doc.md", _chunks())
    store.save("other.md", [])

    with pytest.raises(ValueError, match="was not found in source"):
        store.load_neighborhood(source, chunk_id)


def test_save_separates_chunks_for_distinct_sources(tmp_path):
    store = _store(tmp_path)

    store.save("docs/api/guide.md", [])
    store.save("docs/user/guide.md", _chunks())

    assert store.load("docs/api/guide.md") == []
    assert [chunk.id for chunk in store.load("docs/user/guide.md")] == [
        "doc-chunk-0000",
        "doc-chunk-0001",
        "doc-chunk-0002",
    ]


def test_load_all_returns_chunks_across_sources(tmp_path):
    store = _store(tmp_path)
    store.save("docs/api/guide.md", [_chunks()[0]])
    store.save("docs/user/guide.md", [_chunks()[1]])

    loaded = store.load_all()

    assert {chunk.id for chunk in loaded} == {
        "doc-chunk-0000",
        "doc-chunk-0001",
    }


def test_save_replaces_existing_chunks_for_source(tmp_path):
    store = _store(tmp_path)
    store.save("doc.md", _chunks())
    store.save("doc.md", [_chunks()[0]])

    assert [chunk.id for chunk in store.load("doc.md")] == ["doc-chunk-0000"]


def test_delete_removes_saved_chunks(tmp_path):
    store = _store(tmp_path)
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

    assert store.load("doc.md") == []


def test_delete_is_noop_for_unsaved_source(tmp_path):
    store = _store(tmp_path)

    store.delete("missing.md")

    assert store.load("missing.md") == []


def test_save_roundtrips_page_range(tmp_path):
    store = _store(tmp_path)
    chunk = Chunk(
        id="pdf-chunk-0000",
        source="doc.pdf",
        index=0,
        text="Page text.",
        page_start=3,
        page_end=4,
    )

    store.save("doc.pdf", [chunk])
    loaded = store.load("doc.pdf")

    assert loaded == [chunk]
    assert loaded[0].page_start == 3
    assert loaded[0].page_end == 4
