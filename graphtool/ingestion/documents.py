from pathlib import Path

from graphtool.ingestion.audio import convert_audio_to_markdown
from graphtool.ingestion.pdf import convert_pdf_to_markdown
from graphtool.ingestion.presentation import convert_pptx_to_markdown
from graphtool.llm.base import AudioTranscriptionClient, LLMClient

_AUDIO_SUFFIXES = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}
_SUPPORTED_SUFFIXES = {".md", ".pdf", ".pptx", *_AUDIO_SUFFIXES}


def load_documents(
    directory: str | Path,
    *,
    source_root: str | Path,
    pdf_llm: LLMClient,
    pdf_cache_dir: str | Path,
    presentation_cache_dir: str | Path,
    audio_transcriber: AudioTranscriptionClient,
    audio_cache_dir: str | Path,
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
        elif document_path.suffix.lower() == ".pdf":
            documents[source] = convert_pdf_to_markdown(
                document_path,
                source,
                pdf_llm,
                pdf_cache_dir,
            )
        elif document_path.suffix.lower() == ".pptx":
            documents[source] = convert_pptx_to_markdown(
                document_path,
                source,
                pdf_llm,
                presentation_cache_dir,
                pdf_cache_dir,
            )
        else:
            documents[source] = convert_audio_to_markdown(
                document_path,
                source,
                audio_transcriber,
                audio_cache_dir,
            )
    return documents
