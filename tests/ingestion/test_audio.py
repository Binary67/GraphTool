import json

import pytest

from graphtool.ingestion import audio
from graphtool.ingestion.audio import convert_audio_to_markdown
from graphtool.source import source_key


class FakeTranscriber:
    def __init__(self, responses, *, model="transcription-deployment"):
        self.transcription_model = model
        self.responses = list(responses)
        self.calls = []

    def transcribe_audio(self, path, *, prompt=None):
        self.calls.append((path.read_bytes(), prompt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _prepare(monkeypatch, duration_milliseconds):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        audio,
        "_probe_duration",
        lambda path, source, ffprobe: duration_milliseconds,
    )
    render_calls = []

    def fake_render(
        source_path,
        chunk_path,
        start_milliseconds,
        end_milliseconds,
        ffmpeg,
        source,
    ):
        render_calls.append((start_milliseconds, end_milliseconds))
        chunk_path.write_bytes(
            f"audio-{start_milliseconds}-{end_milliseconds}".encode()
        )

    monkeypatch.setattr(audio, "_render_chunk", fake_render)
    return render_calls


def test_convert_audio_chunks_transcribes_and_assembles_markdown(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 40 * 60 * 1000)
    path = tmp_path / "quarterly-review.mp3"
    path.write_bytes(b"original-audio")
    first_transcript = (
        "Opening facts. Shared boundary words remain exactly the same."
    )
    transcriber = FakeTranscriber(
        [
            first_transcript,
            (
                "Shared boundary words remain exactly the same. "
                "New section facts."
            ),
        ]
    )

    markdown = convert_audio_to_markdown(
        path,
        "documents/recordings/quarterly-review.mp3",
        transcriber,
        tmp_path / "cache",
    )

    assert render_calls == [
        (0, 1_205_000),
        (1_200_000, 2_400_000),
    ]
    assert transcriber.calls[0][1] is None
    assert transcriber.calls[1][1] == first_transcript
    assert markdown == (
        "# Transcript: quarterly-review.mp3\n\n"
        "## 00:00:00\n\n"
        "Opening facts. Shared boundary words remain exactly the same.\n\n"
        "## 00:20:00\n\n"
        "New section facts.\n"
    )


def test_convert_audio_uses_complete_cache_without_external_tools(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    expected = convert_audio_to_markdown(
        path,
        "documents/recordings/recording.mp3",
        FakeTranscriber(["Cached transcript."]),
        cache_dir,
    )
    monkeypatch.setattr(
        audio.shutil,
        "which",
        lambda name: pytest.fail("completed cache looked up external tools"),
    )
    transcriber = FakeTranscriber([])

    actual = convert_audio_to_markdown(
        path,
        "documents/recordings/recording.mp3",
        transcriber,
        cache_dir,
    )

    assert actual == expected
    assert transcriber.calls == []


def test_convert_audio_resumes_completed_chunks_after_failure(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 40 * 60 * 1000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"

    with pytest.raises(RuntimeError, match="request failed"):
        convert_audio_to_markdown(
            path,
            source,
            FakeTranscriber(["First chunk.", RuntimeError("request failed")]),
            cache_dir,
        )

    resumed = FakeTranscriber(["Second chunk."])
    markdown = convert_audio_to_markdown(
        path,
        source,
        resumed,
        cache_dir,
    )

    assert render_calls == [
        (0, 1_205_000),
        (1_200_000, 2_400_000),
        (1_200_000, 2_400_000),
    ]
    assert len(resumed.calls) == 1
    assert resumed.calls[0][1].endswith("First chunk.")
    assert "First chunk." in markdown
    assert "Second chunk." in markdown


def test_convert_audio_invalidates_cache_when_model_changes(monkeypatch, tmp_path):
    render_calls = _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["First."], model="transcribe-a"),
        cache_dir,
    )

    markdown = convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["Second."], model="transcribe-b"),
        cache_dir,
    )

    assert render_calls == [(0, 60_000), (0, 60_000)]
    assert "Second." in markdown
    manifest_path = cache_dir / source_key(source) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"] == "transcribe-b"
    assert manifest["complete"] is True
    assert manifest["markdown_hash"]


def test_convert_audio_fails_clearly_without_ffprobe(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"audio")

    with pytest.raises(RuntimeError, match="ffprobe was not found"):
        convert_audio_to_markdown(
            path,
            "documents/recordings/recording.mp3",
            FakeTranscriber([]),
            tmp_path / "cache",
        )


def test_render_chunk_normalizes_audio_and_checks_size(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        audio.Path(command[-1]).write_bytes(b"normalized-audio")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    output_path = tmp_path / "chunk.mp3"

    audio._render_chunk(
        tmp_path / "source.wav",
        output_path,
        1_000,
        6_000,
        "/usr/bin/ffmpeg",
        "documents/recordings/source.wav",
    )

    command, kwargs = calls[0]
    assert command[:7] == [
        "/usr/bin/ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        "1.000",
        "-i",
    ]
    assert command[command.index("-t") + 1] == "5.000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-b:a") + 1] == "64k"
    assert kwargs == {"check": True, "capture_output": True, "text": True}


def test_chunk_boundaries_do_not_create_tiny_overlap_only_chunk():
    assert audio._chunk_boundaries(1_203_000) == [(0, 1_203_000)]
    assert audio._chunk_boundaries(2_403_000) == [
        (0, 1_205_000),
        (1_200_000, 2_403_000),
    ]
