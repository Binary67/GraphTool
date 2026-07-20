import re
from dataclasses import dataclass

import tiktoken

from graphtool.chunking.types import Chunk
from graphtool.source import source_key

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE_MARKER_PATTERN = re.compile(r"^<!--\s*graphtool:page=(\d+)\s*-->$")
# GPT-5-family text encoding.
_ENCODING = tiktoken.get_encoding("o200k_base")
_TARGET_TOKENS = 4000
_MAX_TOKENS = 8000
_SECTION_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class _Fragment:
    text: str
    heading_path: list[str]
    page_start: int | None = None
    page_end: int | None = None


def chunk_markdown(
    markdown: str,
    source: str,
) -> list[Chunk]:
    sections = _split_sections(markdown)
    fragments = [
        _Fragment(
            text=text,
            heading_path=section.heading_path,
            page_start=section.page_start,
            page_end=section.page_end,
        )
        for section in sections
        for text in _split_text(section.text, _MAX_TOKENS)
        if _has_content(text)
    ]
    packed = _pack_fragments(fragments)
    key = source_key(source)

    return [
        Chunk(
            id=f"{key}-chunk-{index:04d}",
            source=source,
            index=index,
            text=fragment.text,
            heading_path=fragment.heading_path,
            page_start=fragment.page_start,
            page_end=fragment.page_end,
        )
        for index, fragment in enumerate(packed)
    ]


def _pack_fragments(
    fragments: list[_Fragment],
) -> list[_Fragment]:
    packed: list[_Fragment] = []
    current: _Fragment | None = None

    for fragment in fragments:
        if current is None:
            current = fragment
            continue

        combined_text = f"{current.text}{_SECTION_SEPARATOR}{fragment.text}"
        if (
            _token_count(current.text) < _TARGET_TOKENS
            and _token_count(combined_text) <= _MAX_TOKENS
        ):
            current = _Fragment(
                text=combined_text,
                heading_path=_common_heading_path(
                    current.heading_path,
                    fragment.heading_path,
                ),
                page_start=_first_page(current.page_start, fragment.page_start),
                page_end=_last_page(current.page_end, fragment.page_end),
            )
            continue

        packed.append(current)
        current = fragment

    if current is not None:
        packed.append(current)

    return packed


def _common_heading_path(left: list[str], right: list[str]) -> list[str]:
    common = []
    for left_heading, right_heading in zip(left, right):
        if left_heading != right_heading:
            break
        common.append(left_heading)
    return common


def _has_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _split_sections(markdown: str) -> list[_Fragment]:
    sections: list[_Fragment] = []
    heading_stack: list[str] = []
    current_heading_path: list[str] = []
    current_lines: list[str] = []
    current_page: int | None = None

    for line in markdown.splitlines():
        page_marker = _PAGE_MARKER_PATTERN.match(line)
        if page_marker:
            _append_section(
                sections,
                current_lines,
                current_heading_path,
                current_page,
            )
            current_lines = []
            current_page = int(page_marker.group(1))
            continue

        heading = _HEADING_PATTERN.match(line)
        if heading:
            _append_section(
                sections,
                current_lines,
                current_heading_path,
                current_page,
            )

            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading_path = list(heading_stack)
            current_lines = [line]
            continue

        current_lines.append(line)

    _append_section(sections, current_lines, current_heading_path, current_page)
    return sections


def _append_section(
    sections: list[_Fragment],
    lines: list[str],
    heading_path: list[str],
    page: int | None,
) -> None:
    text = "\n".join(lines).strip()
    if text:
        sections.append(
            _Fragment(
                text=text,
                heading_path=list(heading_path),
                page_start=page,
                page_end=page,
            )
        )


def _first_page(left: int | None, right: int | None) -> int | None:
    return left if left is not None else right


def _last_page(left: int | None, right: int | None) -> int | None:
    return right if right is not None else left


def _split_text(text: str, max_tokens: int) -> list[str]:
    if _token_count(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in _split_paragraphs(text):
        if _token_count(paragraph) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_tokens))
            continue

        if not current:
            current = paragraph
        elif _token_count(f"{current}\n\n{paragraph}") <= max_tokens:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    remaining = paragraph.strip()

    while _token_count(remaining) > max_tokens:
        split_at = _split_index(remaining, max_tokens)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _split_index(text: str, max_tokens: int) -> int:
    encoded = _ENCODING.encode(text, disallowed_special=())
    prefix = _ENCODING.decode(encoded[:max_tokens])
    split_limit = min(len(prefix), len(text))
    while _token_count(text[:split_limit]) > max_tokens:
        split_limit -= 1

    whitespace_indexes = [
        match.start() for match in re.finditer(r"\s+", text[: split_limit + 1])
    ]
    if whitespace_indexes:
        split_at = whitespace_indexes[-1]
        if split_at >= split_limit // 2:
            return split_at

    return split_limit


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))
