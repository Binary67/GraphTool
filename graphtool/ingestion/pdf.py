import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMImageContent, LLMMessage, LLMTextContent
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import source_key

PDF_BATCH_MAX_PAGES = 2
PDF_BATCH_MAX_EXTRACTED_CHARS = 16_000
PDF_CONTEXT_TAIL_CHARS = 1_000
PDF_RENDER_DPI = 150
PDF_PROMPT_REVISION = 3
_PAGE_MARKER_TEMPLATE = "<!-- graphtool:page={page_number} -->"
RUN_LOGGER = logging.getLogger(LOGGER_NAME)

_CONVERSION_INSTRUCTIONS = (
    "Convert PDF pages into faithful Markdown for downstream knowledge extraction. "
    "Transcribe rather than summarize or rewrite. Preserve the original reading "
    "order, heading hierarchy, paragraphs, lists, tables, code, footnotes, captions, "
    "and visible links. Omit repeated headers, footers, and printed page numbers. "
    "If a page contains only repeated template content, such as headers, footers, "
    "confidentiality labels, logos, copyright notices, or page numbers, set "
    "is_blank to true and return empty Markdown for that page. Continue processing "
    "the remaining pages and still return one page record for every requested page. "
    "Describe meaningful figures only from clearly visible content, and never infer "
    "unreadable values or facts. Mark unreadable content as [Unclear]. Do not wrap "
    "Markdown in a code fence and do not add page markers; the caller adds them."
)
_SYSTEM_PROMPT = (
    f"{_CONVERSION_INSTRUCTIONS} "
    "Return exactly one page record for every requested page, in request order."
)
_SINGLE_PAGE_SYSTEM_PROMPT = (
    f"{_CONVERSION_INSTRUCTIONS} Return conversion content for exactly the requested "
    "page. Do not return a page number; the caller assigns it."
)


class ConvertedPdfPage(BaseModel):
    page_number: int
    markdown: str
    is_blank: bool = False
    warnings: list[str] = Field(default_factory=list)


class PdfBatchConversion(BaseModel):
    pages: list[ConvertedPdfPage]
    ending_heading_path: list[str] = Field(default_factory=list)


class PdfPageConversion(BaseModel):
    markdown: str
    is_blank: bool = False
    warnings: list[str] = Field(default_factory=list)
    ending_heading_path: list[str] = Field(default_factory=list)


class _PdfConversionManifest(BaseModel):
    source_hash: str
    model: str
    prompt_revision: int
    render_dpi: int
    batch_max_pages: int
    batch_max_extracted_chars: int
    page_count: int
    complete: bool = False
    markdown_hash: str | None = None


def convert_pdf_to_markdown(
    path: str | Path,
    source: str,
    llm: LLMClient,
    cache_dir: str | Path,
) -> str:
    pdf_path = Path(path)
    with pdf_path.open("rb") as pdf_file:
        source_hash = hashlib.file_digest(pdf_file, "sha256").hexdigest()
    source_cache_dir = Path(cache_dir) / source_key(source)
    manifest_path = source_cache_dir / "manifest.json"
    markdown_path = source_cache_dir / "document.md"
    manifest = _load_manifest(manifest_path)
    if manifest is not None and _same_source_and_settings(
        manifest,
        source_hash,
        llm.text_model,
    ):
        if manifest.complete and markdown_path.exists():
            markdown = markdown_path.read_text(encoding="utf-8")
            if _text_hash(markdown) == manifest.markdown_hash:
                return markdown

    page_texts = _extract_page_texts(pdf_path, source)
    expected_manifest = _PdfConversionManifest(
        source_hash=source_hash,
        model=llm.text_model,
        prompt_revision=PDF_PROMPT_REVISION,
        render_dpi=PDF_RENDER_DPI,
        batch_max_pages=PDF_BATCH_MAX_PAGES,
        batch_max_extracted_chars=PDF_BATCH_MAX_EXTRACTED_CHARS,
        page_count=len(page_texts),
    )
    if manifest is None or not _same_conversion(manifest, expected_manifest):
        if source_cache_dir.exists():
            shutil.rmtree(source_cache_dir)
        manifest = None

    source_cache_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = source_cache_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        _write_model_atomic(manifest_path, expected_manifest)

    converted_pages = []
    heading_path: list[str] = []
    markdown_tail = ""
    pdftoppm: str | None = None
    for page_batch in _make_page_batches(page_texts):
        page_numbers = [page_number for page_number, _ in page_batch]
        batch_path = batches_dir / _batch_file_name(page_numbers)
        if batch_path.exists():
            conversion = PdfBatchConversion.model_validate_json(batch_path.read_text())
            conversion = _validate_conversion(conversion, page_numbers, source)
        else:
            if pdftoppm is None:
                pdftoppm = shutil.which("pdftoppm")
                if pdftoppm is None:
                    raise RuntimeError(
                        f"Cannot convert {source!r}: Poppler pdftoppm was not found."
                    )
            page_images = _render_pages(
                pdf_path,
                page_numbers,
                pdftoppm,
                source,
            )
            conversion = _convert_batch_with_recovery(
                page_batch,
                page_images,
                heading_path,
                markdown_tail,
                llm,
                source,
            )
            _write_model_atomic(batch_path, conversion)

        converted_pages.extend(conversion.pages)
        heading_path = conversion.ending_heading_path
        if markdown_tail:
            markdown_tail += "\n"
        markdown_tail = (
            markdown_tail + _assemble_markdown(conversion.pages)
        )[-PDF_CONTEXT_TAIL_CHARS:]

    markdown = _assemble_markdown(converted_pages)
    _write_text_atomic(markdown_path, markdown)
    completed_manifest = expected_manifest.model_copy(
        update={"complete": True, "markdown_hash": _text_hash(markdown)}
    )
    _write_model_atomic(manifest_path, completed_manifest)
    return markdown


def _extract_page_texts(pdf_path: Path, source: str) -> list[str]:
    try:
        reader = PdfReader(pdf_path)
    except (OSError, PdfReadError) as exc:
        raise ValueError(f"Cannot read PDF {source!r}.") from exc

    if reader.is_encrypted:
        raise ValueError(f"Password-protected PDF {source!r} is not supported.")
    if not reader.pages:
        raise ValueError(f"PDF {source!r} has no pages.")

    texts = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            texts.append(page.extract_text() or "")
        except Exception as exc:
            raise ValueError(
                f"Cannot extract text from {source!r} page {page_number}."
            ) from exc
    return texts


def _make_page_batches(page_texts: list[str]) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for page_number, text in enumerate(page_texts, start=1):
        if current and (
            len(current) == PDF_BATCH_MAX_PAGES
            or current_chars + len(text) > PDF_BATCH_MAX_EXTRACTED_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((page_number, text))
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


def _render_pages(
    pdf_path: Path,
    page_numbers: list[int],
    pdftoppm: str,
    source: str,
) -> list[bytes]:
    with tempfile.TemporaryDirectory(prefix="graphtool-pdf-") as temporary_dir:
        prefix = Path(temporary_dir) / "page"
        command = [
            pdftoppm,
            "-f",
            str(page_numbers[0]),
            "-l",
            str(page_numbers[-1]),
            "-r",
            str(PDF_RENDER_DPI),
            "-png",
            str(pdf_path),
            str(prefix),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown Poppler error"
            raise RuntimeError(
                f"Cannot render {source!r} pages {page_numbers[0]}-"
                f"{page_numbers[-1]}: {detail}"
            ) from exc

        image_paths = sorted(Path(temporary_dir).glob("page-*.png"))
        if len(image_paths) != len(page_numbers):
            raise RuntimeError(
                f"Expected {len(page_numbers)} rendered pages for {source!r}, "
                f"received {len(image_paths)}."
            )
        return [image_path.read_bytes() for image_path in image_paths]


def _convert_batch(
    pages: list[tuple[int, str]],
    images: list[bytes],
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
    correction: str | None = None,
) -> PdfBatchConversion:
    context = (
        "Convert the requested pages below. Previous heading path: "
        f"{heading_path or ['None']}. The following previous Markdown is context "
        "only; do not repeat it:\n\n"
        f"{previous_markdown or '[None]'}"
    )
    if correction is not None:
        context = f"{context}\n\nCorrection required:\n{correction}"
    content = [LLMTextContent(text=context)]
    for (page_number, extracted_text), image in zip(pages, images, strict=True):
        content.extend(
            [
                LLMTextContent(
                    text=(
                        f"Page {page_number} extracted text (use as transcription "
                        f"grounding):\n\n{extracted_text or '[No extracted text]'}"
                    )
                ),
                LLMImageContent(data=image, detail="high"),
            ]
        )

    return llm.generate_structured(
        [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=tuple(content)),
        ],
        PdfBatchConversion,
    )


def _convert_batch_with_recovery(
    pages: list[tuple[int, str]],
    images: list[bytes],
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
    source: str,
) -> PdfBatchConversion:
    page_numbers = [page_number for page_number, _ in pages]
    conversion = _convert_batch(
        pages,
        images,
        heading_path,
        previous_markdown,
        llm,
    )
    try:
        conversion = _validate_conversion(conversion, page_numbers, source)
    except ValueError as first_error:
        RUN_LOGGER.warning(
            "Retrying PDF batch conversion source=%s pages=%s error=%s",
            source,
            page_numbers,
            first_error,
        )
        correction = (
            f"The previous response failed validation: {first_error} "
            "Return exactly one page record for each of these page numbers, in this "
            f"exact order: {page_numbers}. Never omit a page record; use is_blank=true "
            "and empty Markdown when a page has no meaningful content."
        )
        conversion = _convert_batch(
            pages,
            images,
            heading_path,
            previous_markdown,
            llm,
            correction,
        )
        try:
            conversion = _validate_conversion(conversion, page_numbers, source)
        except ValueError as retry_error:
            RUN_LOGGER.warning(
                "Falling back to individual PDF pages source=%s pages=%s error=%s",
                source,
                page_numbers,
                retry_error,
            )
            conversion = _convert_pages_individually(
                pages,
                images,
                heading_path,
                previous_markdown,
                llm,
                source,
            )
            RUN_LOGGER.info(
                "Recovered PDF batch conversion with individual pages source=%s "
                "pages=%s",
                source,
                page_numbers,
            )
        else:
            RUN_LOGGER.info(
                "Recovered PDF batch conversion on retry source=%s pages=%s",
                source,
                page_numbers,
            )
    return conversion


def _convert_pages_individually(
    pages: list[tuple[int, str]],
    images: list[bytes],
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
    source: str,
) -> PdfBatchConversion:
    converted_pages = []
    current_heading_path = list(heading_path)
    markdown_tail = previous_markdown
    for (page_number, extracted_text), image in zip(pages, images, strict=True):
        page_conversion = _convert_page(
            page_number,
            extracted_text,
            image,
            current_heading_path,
            markdown_tail,
            llm,
        )
        conversion = _validate_conversion(
            PdfBatchConversion(
                pages=[
                    ConvertedPdfPage(
                        page_number=page_number,
                        markdown=page_conversion.markdown,
                        is_blank=page_conversion.is_blank,
                        warnings=page_conversion.warnings,
                    )
                ],
                ending_heading_path=page_conversion.ending_heading_path,
            ),
            [page_number],
            source,
        )
        converted_pages.extend(conversion.pages)
        current_heading_path = conversion.ending_heading_path
        if markdown_tail:
            markdown_tail += "\n"
        markdown_tail = (
            markdown_tail + _assemble_markdown(conversion.pages)
        )[-PDF_CONTEXT_TAIL_CHARS:]
    return PdfBatchConversion(
        pages=converted_pages,
        ending_heading_path=current_heading_path,
    )


def _convert_page(
    page_number: int,
    extracted_text: str,
    image: bytes,
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
) -> PdfPageConversion:
    context = (
        f"Convert page {page_number}. Previous heading path: "
        f"{heading_path or ['None']}. The following previous Markdown is context "
        "only; do not repeat it:\n\n"
        f"{previous_markdown or '[None]'}"
    )
    return llm.generate_structured(
        [
            LLMMessage(role="system", content=_SINGLE_PAGE_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    LLMTextContent(text=context),
                    LLMTextContent(
                        text=(
                            f"Page {page_number} extracted text (use as transcription "
                            f"grounding):\n\n{extracted_text or '[No extracted text]'}"
                        )
                    ),
                    LLMImageContent(data=image, detail="high"),
                ),
            ),
        ],
        PdfPageConversion,
    )


def _validate_conversion(
    conversion: PdfBatchConversion,
    expected_page_numbers: list[int],
    source: str,
) -> PdfBatchConversion:
    actual_page_numbers = [page.page_number for page in conversion.pages]
    if actual_page_numbers != expected_page_numbers:
        raise ValueError(
            f"PDF conversion for {source!r} expected pages "
            f"{expected_page_numbers}, received {actual_page_numbers}."
        )

    normalized_pages = []
    for page in conversion.pages:
        markdown = _normalize_markdown(page.markdown)
        if page.is_blank:
            markdown = ""
        elif not markdown:
            raise ValueError(
                f"PDF conversion for {source!r} page {page.page_number} returned "
                "empty Markdown without marking the page blank."
            )
        normalized_pages.append(page.model_copy(update={"markdown": markdown}))

    return conversion.model_copy(update={"pages": normalized_pages})


def _assemble_markdown(pages: list[ConvertedPdfPage]) -> str:
    blocks = []
    for page in pages:
        marker = _PAGE_MARKER_TEMPLATE.format(page_number=page.page_number)
        blocks.append(f"{marker}\n\n{page.markdown}".rstrip())
    return "\n\n".join(blocks).rstrip() + "\n"


def _normalize_markdown(markdown: str) -> str:
    return markdown.replace("\r\n", "\n").replace("\r", "\n").strip()


def _batch_file_name(page_numbers: list[int]) -> str:
    return f"pages-{page_numbers[0]:05d}-{page_numbers[-1]:05d}.json"


def _same_conversion(
    current: _PdfConversionManifest,
    expected: _PdfConversionManifest,
) -> bool:
    fields = (
        "source_hash",
        "model",
        "prompt_revision",
        "render_dpi",
        "batch_max_pages",
        "batch_max_extracted_chars",
        "page_count",
    )
    return all(getattr(current, field) == getattr(expected, field) for field in fields)


def _same_source_and_settings(
    manifest: _PdfConversionManifest,
    source_hash: str,
    model: str,
) -> bool:
    return (
        manifest.source_hash == source_hash
        and manifest.model == model
        and manifest.prompt_revision == PDF_PROMPT_REVISION
        and manifest.render_dpi == PDF_RENDER_DPI
        and manifest.batch_max_pages == PDF_BATCH_MAX_PAGES
        and manifest.batch_max_extracted_chars == PDF_BATCH_MAX_EXTRACTED_CHARS
    )


def _load_manifest(path: Path) -> _PdfConversionManifest | None:
    if not path.exists():
        return None
    return _PdfConversionManifest.model_validate_json(path.read_text())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    _write_text_atomic(path, model.model_dump_json(indent=2))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
