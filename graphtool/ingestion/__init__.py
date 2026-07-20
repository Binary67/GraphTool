"""Document discovery and normalization."""

from graphtool.ingestion.audio import convert_audio_to_markdown
from graphtool.ingestion.documents import load_documents
from graphtool.ingestion.pdf import convert_pdf_to_markdown

__all__ = [
    "convert_audio_to_markdown",
    "convert_pdf_to_markdown",
    "load_documents",
]
