import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from graphtool.ingestion.pdf import convert_pdf_to_markdown
from graphtool.llm.base import LLMClient
from graphtool.source import source_key

PRESENTATION_CONVERSION_REVISION = 1


class _PresentationConversionManifest(BaseModel):
    source_hash: str
    conversion_revision: int
    page_count: int
    complete: bool = False
    pdf_hash: str | None = None


def convert_pptx_to_markdown(
    path: str | Path,
    source: str,
    llm: LLMClient,
    presentation_cache_dir: str | Path,
    pdf_cache_dir: str | Path,
) -> str:
    pdf_path = convert_pptx_to_pdf(path, source, presentation_cache_dir)
    return convert_pdf_to_markdown(pdf_path, source, llm, pdf_cache_dir)


def convert_pptx_to_pdf(
    path: str | Path,
    source: str,
    cache_dir: str | Path,
) -> Path:
    presentation_path = Path(path)
    with presentation_path.open("rb") as presentation_file:
        source_hash = hashlib.file_digest(presentation_file, "sha256").hexdigest()

    source_cache_dir = Path(cache_dir) / source_key(source)
    manifest_path = source_cache_dir / "manifest.json"
    pdf_path = source_cache_dir / "presentation.pdf"
    manifest = _load_manifest(manifest_path)
    if (
        manifest is not None
        and manifest.source_hash == source_hash
        and manifest.conversion_revision == PRESENTATION_CONVERSION_REVISION
        and manifest.complete
        and pdf_path.exists()
        and _file_hash(pdf_path) == manifest.pdf_hash
    ):
        return pdf_path

    soffice = shutil.which("soffice")
    if soffice is None:
        raise RuntimeError(
            f"Cannot convert {source!r}: LibreOffice soffice was not found."
        )

    with tempfile.TemporaryDirectory(prefix="graphtool-pptx-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        command = [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temporary_path),
            str(presentation_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (
                exc.stderr.strip()
                or exc.stdout.strip()
                or "unknown LibreOffice error"
            )
            raise RuntimeError(f"Cannot convert {source!r} to PDF: {detail}") from exc

        converted_pdf_path = temporary_path / f"{presentation_path.stem}.pdf"
        if not converted_pdf_path.exists():
            raise RuntimeError(
                f"Cannot convert {source!r} to PDF: LibreOffice produced no PDF."
            )
        page_count = _validate_pdf(converted_pdf_path, source)
        pdf_hash = _file_hash(converted_pdf_path)

        source_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_pdf_path = pdf_path.with_suffix(".pdf.tmp")
        shutil.copyfile(converted_pdf_path, temporary_pdf_path)
        temporary_pdf_path.replace(pdf_path)

    completed_manifest = _PresentationConversionManifest(
        source_hash=source_hash,
        conversion_revision=PRESENTATION_CONVERSION_REVISION,
        page_count=page_count,
        complete=True,
        pdf_hash=pdf_hash,
    )
    _write_model_atomic(manifest_path, completed_manifest)
    return pdf_path


def _validate_pdf(path: Path, source: str) -> int:
    try:
        reader = PdfReader(path)
    except (OSError, PdfReadError) as exc:
        raise RuntimeError(
            f"Cannot convert {source!r} to PDF: generated PDF is unreadable."
        ) from exc
    if reader.is_encrypted:
        raise RuntimeError(
            f"Cannot convert {source!r} to PDF: generated PDF is encrypted."
        )
    if not reader.pages:
        raise RuntimeError(
            f"Cannot convert {source!r} to PDF: generated PDF has no pages."
        )
    return len(reader.pages)


def _load_manifest(path: Path) -> _PresentationConversionManifest | None:
    if not path.exists():
        return None
    return _PresentationConversionManifest.model_validate_json(path.read_text())


def _file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)
