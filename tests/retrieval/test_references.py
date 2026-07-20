from graphtool.retrieval import (
    SourceReference,
    format_source_location,
    format_source_reference,
)


def test_format_powerpoint_source_reference_uses_slide_labels():
    assert format_source_reference(
        SourceReference(
            source="documents/slides.pptx",
            page_start=2,
            page_end=2,
        )
    ) == "documents/slides.pptx (slide 2)"
    assert (
        format_source_location("documents/slides.pptx", 2, 4) == "slides 2-4"
    )


def test_format_pdf_source_reference_keeps_page_labels():
    assert format_source_reference(
        SourceReference(
            source="documents/manual.pdf",
            page_start=2,
            page_end=4,
        )
    ) == "documents/manual.pdf (pp. 2-4)"
