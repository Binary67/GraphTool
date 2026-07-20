from pathlib import PurePosixPath

from graphtool.retrieval.types import SourceReference


def format_source_reference(reference: SourceReference) -> str:
    location = format_source_location(
        reference.source,
        reference.page_start,
        reference.page_end,
    )
    if not location:
        return reference.source
    return f"{reference.source} ({location})"


def format_source_location(
    source: str,
    page_start: int | None,
    page_end: int | None,
) -> str:
    if page_start is None:
        return ""
    is_presentation = PurePosixPath(source).suffix.lower() == ".pptx"
    if page_start == page_end:
        label = "slide" if is_presentation else "p."
        return f"{label} {page_start}"
    label = "slides" if is_presentation else "pp."
    return f"{label} {page_start}-{page_end}"
