from graphtool.chunking.markdown import chunk_markdown
from graphtool.source import source_key


def test_chunk_markdown_merges_neighboring_sections_and_preserves_headings():
    markdown = "# Guide\nIntro.\n\n## Install\nRun setup.\n\n## Use\nStart."

    chunks = chunk_markdown(markdown, "docs/guide.md")
    key = source_key("docs/guide.md")

    assert [chunk.id for chunk in chunks] == [f"{key}-chunk-0000"]
    assert [chunk.index for chunk in chunks] == [0]
    assert [chunk.source for chunk in chunks] == ["docs/guide.md"]
    assert chunks[0].heading_path == ["Guide"]
    assert chunks[0].text == markdown


def test_chunk_markdown_returns_empty_list_for_empty_markdown():
    assert chunk_markdown("", "empty.md") == []
    assert chunk_markdown("  \n\n", "empty.md") == []


def test_chunk_markdown_closes_chunk_after_crossing_soft_target():
    markdown = (
        f"# Guide\n{'a' * 2000}\n\n"
        f"## Details\n{'b' * 1500}\n\n"
        "## Next\nShort."
    )

    chunks = chunk_markdown(markdown, "guide.md")

    assert len(chunks) == 2
    assert 3000 < len(chunks[0].text) <= 6000
    assert "## Details" in chunks[0].text
    assert "## Next" not in chunks[0].text
    assert chunks[0].heading_path == ["Guide"]
    assert chunks[1].heading_path == ["Guide", "Next"]


def test_chunk_markdown_does_not_merge_past_hard_ceiling():
    markdown = f"# First\n{'a' * 2900}\n\n# Second\n{'b' * 3200}"

    chunks = chunk_markdown(markdown, "separate.md")

    assert len(chunks) == 2
    assert chunks[0].heading_path == ["First"]
    assert chunks[1].heading_path == ["Second"]
    assert all(len(chunk.text) <= 6000 for chunk in chunks)


def test_chunk_markdown_splits_oversized_sections_by_paragraph():
    first_paragraph = "alpha " * 700
    second_paragraph = "beta " * 700
    markdown = f"# Notes\n{first_paragraph}\n\n{second_paragraph}"

    chunks = chunk_markdown(markdown, "notes.md")

    assert len(chunks) == 2
    assert all(len(chunk.text) <= 6000 for chunk in chunks)
    assert all(chunk.heading_path == ["Notes"] for chunk in chunks)
    assert chunks[0].text.startswith("# Notes")
    assert chunks[1].text.startswith("beta")


def test_chunk_markdown_splits_single_oversized_paragraph_at_whitespace():
    markdown = "# Long\n" + "word " * 1500

    chunks = chunk_markdown(markdown, "long.md")

    assert len(chunks) == 2
    assert all(len(chunk.text) <= 6000 for chunk in chunks)
    assert all(chunk.text == chunk.text.strip() for chunk in chunks)
    assert chunks[0].text.startswith("# Long")


def test_chunk_markdown_uses_common_heading_path_for_unrelated_sections():
    chunks = chunk_markdown("# First\nOne.\n\n# Second\nTwo.", "headings.md")

    assert len(chunks) == 1
    assert chunks[0].heading_path == []


def test_chunk_markdown_omits_punctuation_only_fragments():
    chunks = chunk_markdown("|\n\n# Useful\nFact.", "fragments.md")

    assert len(chunks) == 1
    assert chunks[0].text == "# Useful\nFact."
    assert chunks[0].index == 0


def test_chunk_markdown_uses_source_path_in_chunk_ids():
    first = chunk_markdown("# Guide\nFirst.", "docs/api/guide.md")
    second = chunk_markdown("# Guide\nSecond.", "docs/user/guide.md")

    assert first[0].id != second[0].id


def test_chunk_markdown_consumes_page_markers_and_tracks_page_range():
    markdown = (
        "<!-- graphtool:page=3 -->\n\n"
        "# Guide\nFirst page.\n\n"
        "<!-- graphtool:page=4 -->\n\n"
        "Second page."
    )

    chunks = chunk_markdown(markdown, "documents/guide.pdf")

    assert len(chunks) == 1
    assert chunks[0].text == "# Guide\nFirst page.\n\nSecond page."
    assert chunks[0].heading_path == ["Guide"]
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 4


def test_chunk_markdown_leaves_pages_unset_for_markdown_sources():
    chunks = chunk_markdown("# Guide\nText.", "documents/guide.md")

    assert chunks[0].page_start is None
    assert chunks[0].page_end is None


def test_chunk_markdown_skips_empty_page_marker():
    chunks = chunk_markdown(
        "<!-- graphtool:page=1 -->\n\n"
        "<!-- graphtool:page=2 -->\n\n# Guide\nText.\n",
        "documents/slides.pptx",
    )

    assert len(chunks) == 1
    assert chunks[0].text == "# Guide\nText."
    assert chunks[0].page_start == 2
    assert chunks[0].page_end == 2
