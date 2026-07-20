import json
import subprocess
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from graphtool.ingestion import presentation
from graphtool.source import source_key


def _fake_soffice(monkeypatch, calls, *, page_count=2):
    monkeypatch.setattr(
        presentation.shutil,
        "which",
        lambda name: "/usr/bin/soffice",
    )

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        output_dir = Path(command[command.index("--outdir") + 1])
        input_path = Path(command[-1])
        writer = PdfWriter()
        for _ in range(page_count):
            writer.add_blank_page(width=640, height=360)
        with (output_dir / f"{input_path.stem}.pdf").open("wb") as file:
            writer.write(file)

    monkeypatch.setattr(presentation.subprocess, "run", fake_run)


def test_convert_pptx_to_pdf_caches_by_source_hash(monkeypatch, tmp_path):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"first")
    cache_dir = tmp_path / "cache"
    source = "documents/slides.pptx"
    calls = []
    _fake_soffice(monkeypatch, calls)

    first = presentation.convert_pptx_to_pdf(path, source, cache_dir)
    second = presentation.convert_pptx_to_pdf(path, source, cache_dir)

    assert first == second
    assert len(calls) == 1
    assert len(PdfReader(first).pages) == 2
    manifest_path = cache_dir / source_key(source) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] is True
    assert manifest["page_count"] == 2
    assert manifest["pdf_hash"]

    path.write_bytes(b"changed")
    presentation.convert_pptx_to_pdf(path, source, cache_dir)

    assert len(calls) == 2


def test_convert_pptx_to_pdf_fails_without_libreoffice(monkeypatch, tmp_path):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"pptx")
    monkeypatch.setattr(presentation.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="LibreOffice soffice was not found"):
        presentation.convert_pptx_to_pdf(
            path,
            "documents/slides.pptx",
            tmp_path / "cache",
        )


def test_convert_pptx_to_pdf_reports_libreoffice_failure(monkeypatch, tmp_path):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"pptx")
    monkeypatch.setattr(
        presentation.shutil,
        "which",
        lambda name: "/usr/bin/soffice",
    )

    def fail_run(command, *, check, capture_output, text):
        raise subprocess.CalledProcessError(1, command, stderr="bad presentation")

    monkeypatch.setattr(presentation.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="bad presentation"):
        presentation.convert_pptx_to_pdf(
            path,
            "documents/slides.pptx",
            tmp_path / "cache",
        )


def test_convert_pptx_to_markdown_preserves_original_source(monkeypatch, tmp_path):
    generated_pdf = tmp_path / "presentation.pdf"
    llm = object()
    calls = []
    monkeypatch.setattr(
        presentation,
        "convert_pptx_to_pdf",
        lambda path, source, cache_dir: generated_pdf,
    )

    def fake_convert_pdf(path, source, pdf_llm, cache_dir):
        calls.append((path, source, pdf_llm, cache_dir))
        return "# Slides\n"

    monkeypatch.setattr(presentation, "convert_pdf_to_markdown", fake_convert_pdf)

    markdown = presentation.convert_pptx_to_markdown(
        tmp_path / "slides.pptx",
        "documents/slides.pptx",
        llm,
        tmp_path / "presentation-cache",
        tmp_path / "pdf-cache",
    )

    assert markdown == "# Slides\n"
    assert calls == [
        (
            generated_pdf,
            "documents/slides.pptx",
            llm,
            tmp_path / "pdf-cache",
        )
    ]
