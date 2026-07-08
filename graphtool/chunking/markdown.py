import re

from graphtool.chunking.types import Chunk
from graphtool.source import source_key

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown(
    markdown: str,
    source: str,
    max_chars: int = 3000,
) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    sections = _split_sections(markdown)
    chunks: list[Chunk] = []
    key = source_key(source)

    for section_text, heading_path in sections:
        for text in _split_text(section_text, max_chars):
            chunks.append(
                Chunk(
                    id=f"{key}-chunk-{len(chunks):04d}",
                    source=source,
                    index=len(chunks),
                    text=text,
                    heading_path=heading_path,
                )
            )

    return chunks


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
