from pathlib import Path

from graphtool.ingestion.pdf import convert_pdf_to_markdown
from graphtool.llm.base import LLMClient

_SUPPORTED_SUFFIXES = {".md", ".pdf"}


def load_documents(
    directory: str | Path,
    *,
    source_root: str | Path,
    pdf_llm: LLMClient,
    pdf_cache_dir: str | Path,
) -> dict[str, str]:
    path = Path(directory)
    if not path.exists():
        return {}

    root = Path(source_root)
    documents = {}
    document_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    for document_path in document_paths:
        source = document_path.relative_to(root).as_posix()
        if document_path.suffix.lower() == ".md":
            documents[source] = document_path.read_text(encoding="utf-8")
        else:
            documents[source] = convert_pdf_to_markdown(
                document_path,
                source,
                pdf_llm,
                pdf_cache_dir,
            )
    return documents
