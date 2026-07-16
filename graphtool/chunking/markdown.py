import re

from graphtool.chunking.types import Chunk
from graphtool.source import source_key

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TARGET_CHARS = 3000
_MAX_CHARS = 6000
_SECTION_SEPARATOR = "\n\n"


def chunk_markdown(
    markdown: str,
    source: str,
) -> list[Chunk]:
    sections = _split_sections(markdown)
    fragments = [
        (text, heading_path)
        for section_text, heading_path in sections
        for text in _split_text(section_text, _MAX_CHARS)
        if _has_content(text)
    ]
    packed = _pack_fragments(fragments)
    key = source_key(source)

    return [
        Chunk(
            id=f"{key}-chunk-{index:04d}",
            source=source,
            index=index,
            text=text,
            heading_path=heading_path,
        )
        for index, (text, heading_path) in enumerate(packed)
    ]


def _pack_fragments(
    fragments: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    packed: list[tuple[str, list[str]]] = []
    current_text = ""
    current_heading_path: list[str] = []

    for text, heading_path in fragments:
        if not current_text:
            current_text = text
            current_heading_path = list(heading_path)
            continue

        combined_text = f"{current_text}{_SECTION_SEPARATOR}{text}"
        if len(current_text) < _TARGET_CHARS and len(combined_text) <= _MAX_CHARS:
            current_text = combined_text
            current_heading_path = _common_heading_path(
                current_heading_path,
                heading_path,
            )
            continue

        packed.append((current_text, current_heading_path))
        current_text = text
        current_heading_path = list(heading_path)

    if current_text:
        packed.append((current_text, current_heading_path))

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


def _split_sections(markdown: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[str] = []
    current_heading_path: list[str] = []
    current_lines: list[str] = []

    for line in markdown.splitlines():
        heading = _HEADING_PATTERN.match(line)
        if heading:
            _append_section(sections, current_lines, current_heading_path)

            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading_path = list(heading_stack)
            current_lines = [line]
            continue

        current_lines.append(line)

    _append_section(sections, current_lines, current_heading_path)
    return sections


def _append_section(
    sections: list[tuple[str, list[str]]],
    lines: list[str],
    heading_path: list[str],
) -> None:
    text = "\n".join(lines).strip()
    if text:
        sections.append((text, list(heading_path)))


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in _split_paragraphs(text):
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars))
            continue

        if not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_chars:
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


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = paragraph.strip()

    while len(remaining) > max_chars:
        split_at = _split_index(remaining, max_chars)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _split_index(text: str, max_chars: int) -> int:
    whitespace_indexes = [
        match.start() for match in re.finditer(r"\s+", text[: max_chars + 1])
    ]
    if whitespace_indexes:
        split_at = whitespace_indexes[-1]
        if split_at >= max_chars // 2:
            return split_at

    return max_chars
