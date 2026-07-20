import pytest

from graphtool.ingestion.documents import load_documents


def test_load_documents_returns_empty_for_missing_directory(tmp_path):
    documents = load_documents(
        tmp_path / "missing",
        source_root=tmp_path,
        pdf_llm=object(),
        pdf_cache_dir=tmp_path / "cache",
        audio_transcriber=object(),
        audio_cache_dir=tmp_path / "audio-cache",
    )

    assert documents == {}


def test_load_documents_reads_markdown_and_converts_pdf(monkeypatch, tmp_path):
    documents_dir = tmp_path / "documents"
    nested_dir = documents_dir / "guides"
    nested_dir.mkdir(parents=True)
    (documents_dir / "ignored.txt").write_text("ignored")
    (documents_dir / "a.MD").write_text("# A")
    pdf_path = nested_dir / "manual.PDF"
    pdf_path.write_bytes(b"pdf")
    convert_calls = []

    def fake_convert(path, source, llm, cache_dir):
        convert_calls.append((path, source, llm, cache_dir))
        return "# Manual"

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_pdf_to_markdown",
        fake_convert,
    )
    llm = object()
    cache_dir = tmp_path / "cache"

    documents = load_documents(
        documents_dir,
        source_root=tmp_path,
        pdf_llm=llm,
        pdf_cache_dir=cache_dir,
        audio_transcriber=object(),
        audio_cache_dir=tmp_path / "audio-cache",
    )

    assert documents == {
        "documents/a.MD": "# A",
        "documents/guides/manual.PDF": "# Manual",
    }
    assert convert_calls == [
        (pdf_path, "documents/guides/manual.PDF", llm, cache_dir)
    ]


def test_load_documents_does_not_return_partial_results_on_pdf_failure(
    monkeypatch,
    tmp_path,
):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide.md").write_text("# Guide")
    (documents_dir / "manual.pdf").write_bytes(b"pdf")

    def fail_conversion(path, source, llm, cache_dir):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_pdf_to_markdown",
        fail_conversion,
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        load_documents(
            documents_dir,
            source_root=tmp_path,
            pdf_llm=object(),
            pdf_cache_dir=tmp_path / "cache",
            audio_transcriber=object(),
            audio_cache_dir=tmp_path / "audio-cache",
        )


def test_load_documents_converts_nested_audio(monkeypatch, tmp_path):
    documents_dir = tmp_path / "documents"
    recordings_dir = documents_dir / "recordings" / "interviews"
    recordings_dir.mkdir(parents=True)
    audio_path = recordings_dir / "customer.MP3"
    audio_path.write_bytes(b"audio")
    calls = []

    def fake_convert(path, source, transcriber, cache_dir):
        calls.append((path, source, transcriber, cache_dir))
        return "# Transcript: customer.MP3\n\nInterview text.\n"

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_audio_to_markdown",
        fake_convert,
    )
    transcriber = object()
    cache_dir = tmp_path / "audio-cache"

    documents = load_documents(
        documents_dir,
        source_root=tmp_path,
        pdf_llm=object(),
        pdf_cache_dir=tmp_path / "pdf-cache",
        audio_transcriber=transcriber,
        audio_cache_dir=cache_dir,
    )

    assert documents == {
        "documents/recordings/interviews/customer.MP3": (
            "# Transcript: customer.MP3\n\nInterview text.\n"
        )
    }
    assert calls == [
        (
            audio_path,
            "documents/recordings/interviews/customer.MP3",
            transcriber,
            cache_dir,
        )
    ]
