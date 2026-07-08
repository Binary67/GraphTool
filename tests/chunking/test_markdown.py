import pytest

from graphtool.chunking.markdown import chunk_markdown


def test_chunk_markdown_splits_by_headings_and_tracks_heading_path():
    markdown = "# Guide\nIntro.\n\n## Install\nRun setup.\n\n## Use\nStart."

    chunks = chunk_markdown(markdown, "docs/guide.md")

    assert [chunk.id for chunk in chunks] == [
        "guide-chunk-0000",
        "guide-chunk-0001",
        "guide-chunk-0002",
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.source for chunk in chunks] == [
        "docs/guide.md",
        "docs/guide.md",
        "docs/guide.md",
    ]
    assert chunks[0].heading_path == ["Guide"]
    assert chunks[1].heading_path == ["Guide", "Install"]
    assert chunks[2].heading_path == ["Guide", "Use"]
    assert chunks[0].text.startswith("# Guide")
    assert chunks[1].text.startswith("## Install")


def test_chunk_markdown_returns_empty_list_for_empty_markdown():
    assert chunk_markdown("", "empty.md") == []
    assert chunk_markdown("  \n\n", "empty.md") == []


def test_chunk_markdown_splits_oversized_sections_by_paragraph():
    markdown = "# Notes\nAlpha alpha alpha.\n\nBeta beta beta.\n\nGamma gamma gamma."

    chunks = chunk_markdown(markdown, "notes.md", max_chars=30)

    assert len(chunks) == 3
    assert [chunk.id for chunk in chunks] == [
        "notes-chunk-0000",
        "notes-chunk-0001",
        "notes-chunk-0002",
    ]
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert all(chunk.heading_path == ["Notes"] for chunk in chunks)


def test_chunk_markdown_splits_single_oversized_paragraph_at_whitespace():
    markdown = "# Long\n" + "word " * 20

    chunks = chunk_markdown(markdown, "long.md", max_chars=25)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 25 for chunk in chunks)
    assert all(chunk.text == chunk.text.strip() for chunk in chunks)
    assert chunks[0].text.startswith("# Long")


def test_chunk_markdown_requires_positive_max_chars():
    with pytest.raises(ValueError, match="max_chars"):
        chunk_markdown("# Title", "doc.md", max_chars=0)
