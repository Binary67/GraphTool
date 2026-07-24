import hashlib
import logging
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel

from graphtool.llm.base import AudioTranscriptionClient
from graphtool.llm.types import AudioTranscript, AudioTranscriptSegment
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import source_key

AUDIO_CHUNK_MILLISECONDS = 20 * 60 * 1000
AUDIO_OVERLAP_MILLISECONDS = 5 * 1000
AUDIO_SAMPLE_RATE = 16_000
AUDIO_BITRATE = "64k"
AUDIO_MAX_CHUNK_BYTES = 24_000_000
AUDIO_CONTEXT_TAIL_CHARS = 500
AUDIO_FORMAT_REVISION = 2
AUDIO_TRANSCRIPTION_MAX_ATTEMPTS = 3
AUDIO_TRANSCRIPTION_BACKOFF_SECONDS = (2.0, 4.0, 8.0)
RUN_LOGGER = logging.getLogger(LOGGER_NAME)


class AudioTranscriptChunk(BaseModel):
    index: int
    start_milliseconds: int
    end_milliseconds: int
    text: str
    segments: list[AudioTranscriptSegment] = []


class _AudioConversionManifest(BaseModel):
    source_hash: str
    model: str
    format_revision: int
    chunk_milliseconds: int
    overlap_milliseconds: int
    sample_rate: int
    bitrate: str
    duration_milliseconds: int
    chunk_count: int
    complete: bool = False
    markdown_hash: str | None = None


def convert_audio_to_markdown(
    path: str | Path,
    source: str,
    transcriber: AudioTranscriptionClient,
    cache_dir: str | Path,
) -> str:
    audio_path = Path(path)
    with audio_path.open("rb") as audio_file:
        source_hash = hashlib.file_digest(audio_file, "sha256").hexdigest()

    source_cache_dir = Path(cache_dir) / source_key(source)
    manifest_path = source_cache_dir / "manifest.json"
    markdown_path = source_cache_dir / "document.md"
    manifest = _load_manifest(manifest_path)
    if manifest is not None and _same_source_and_settings(
        manifest,
        source_hash,
        transcriber.transcription_model,
    ):
        if manifest.complete and markdown_path.exists():
            markdown = markdown_path.read_text(encoding="utf-8")
            if _text_hash(markdown) == manifest.markdown_hash:
                return markdown

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(f"Cannot transcribe {source!r}: ffprobe was not found.")
    duration_milliseconds = _probe_duration(audio_path, source, ffprobe)
    boundaries = _chunk_boundaries(duration_milliseconds)
    expected_manifest = _AudioConversionManifest(
        source_hash=source_hash,
        model=transcriber.transcription_model,
        format_revision=AUDIO_FORMAT_REVISION,
        chunk_milliseconds=AUDIO_CHUNK_MILLISECONDS,
        overlap_milliseconds=AUDIO_OVERLAP_MILLISECONDS,
        sample_rate=AUDIO_SAMPLE_RATE,
        bitrate=AUDIO_BITRATE,
        duration_milliseconds=duration_milliseconds,
        chunk_count=len(boundaries),
    )
    if manifest is None or not _same_conversion(manifest, expected_manifest):
        if source_cache_dir.exists():
            shutil.rmtree(source_cache_dir)
        manifest = None

    source_cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = source_cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        _write_model_atomic(manifest_path, expected_manifest)

    ffmpeg: str | None = None
    chunks: list[AudioTranscriptChunk] = []
    with tempfile.TemporaryDirectory(prefix="graphtool-audio-") as temporary_dir:
        for index, (start_milliseconds, end_milliseconds) in enumerate(boundaries):
            chunk_cache_path = chunks_dir / f"{index:05d}.json"
            if chunk_cache_path.exists():
                chunk = AudioTranscriptChunk.model_validate_json(
                    chunk_cache_path.read_text(encoding="utf-8")
                )
                chunk = _validate_chunk(
                    chunk,
                    index,
                    start_milliseconds,
                    end_milliseconds,
                    source,
                )
            else:
                if ffmpeg is None:
                    ffmpeg = shutil.which("ffmpeg")
                    if ffmpeg is None:
                        raise RuntimeError(
                            f"Cannot transcribe {source!r}: ffmpeg was not found."
                        )
                chunk_path = Path(temporary_dir) / f"{index:05d}.mp3"
                _render_chunk(
                    audio_path,
                    chunk_path,
                    start_milliseconds,
                    end_milliseconds,
                    ffmpeg,
                    source,
                )
                prompt = _context_prompt(chunks[-1].text if chunks else None)
                transcript = _transcribe_with_retry(
                    transcriber,
                    chunk_path,
                    prompt,
                    source,
                    index,
                )
                if not transcript.text:
                    raise ValueError(
                        f"Audio transcription for {source!r} chunk {index} was empty."
                    )
                filtered_segments = _drop_overlap_segments(transcript, index)
                chunk = AudioTranscriptChunk(
                    index=index,
                    start_milliseconds=start_milliseconds,
                    end_milliseconds=end_milliseconds,
                    text=transcript.text,
                    segments=filtered_segments,
                )
                _write_model_atomic(chunk_cache_path, chunk)
            chunks.append(chunk)

    markdown = _assemble_markdown(audio_path.name, chunks)
    _write_text_atomic(markdown_path, markdown)
    completed_manifest = expected_manifest.model_copy(
        update={"complete": True, "markdown_hash": _text_hash(markdown)}
    )
    _write_model_atomic(manifest_path, completed_manifest)
    return markdown


def _probe_duration(audio_path: Path, source: str, ffprobe: str) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        detail = getattr(exc, "stderr", "").strip() or "invalid duration"
        raise ValueError(f"Cannot read audio {source!r}: {detail}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Cannot read audio {source!r}: invalid duration")
    return math.ceil(duration * 1000)


def _chunk_boundaries(duration_milliseconds: int) -> list[tuple[int, int]]:
    boundaries = []
    start = 0
    while start < duration_milliseconds:
        end = min(
            start + AUDIO_CHUNK_MILLISECONDS + AUDIO_OVERLAP_MILLISECONDS,
            duration_milliseconds,
        )
        boundaries.append((start, end))
        if end == duration_milliseconds:
            break
        start += AUDIO_CHUNK_MILLISECONDS
    return boundaries


def _render_chunk(
    source_path: Path,
    chunk_path: Path,
    start_milliseconds: int,
    end_milliseconds: int,
    ffmpeg: str,
    source: str,
) -> None:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-ss",
        _ffmpeg_time(start_milliseconds),
        "-i",
        str(source_path),
        "-t",
        _ffmpeg_time(end_milliseconds - start_milliseconds),
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-b:a",
        AUDIO_BITRATE,
        str(chunk_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Cannot prepare audio {source!r}: {detail}") from exc
    if not chunk_path.exists() or chunk_path.stat().st_size == 0:
        raise RuntimeError(f"Cannot prepare audio {source!r}: ffmpeg produced no audio.")
    if chunk_path.stat().st_size > AUDIO_MAX_CHUNK_BYTES:
        raise ValueError(
            f"Prepared audio chunk for {source!r} exceeds "
            f"{AUDIO_MAX_CHUNK_BYTES} bytes."
        )


def _context_prompt(previous_text: str | None) -> str | None:
    if previous_text is None:
        return None
    return previous_text[-AUDIO_CONTEXT_TAIL_CHARS:]


def _assemble_markdown(
    file_name: str,
    chunks: list[AudioTranscriptChunk],
) -> str:
    blocks = [f"# Transcript: {file_name}"]
    for chunk in chunks:
        text = _segment_text_after_overlap(chunk)
        if text:
            blocks.append(
                f"## {_format_timestamp(chunk.start_milliseconds)}\n\n{text}"
            )
    return "\n\n".join(blocks).rstrip() + "\n"


def _segment_text_after_overlap(chunk: AudioTranscriptChunk) -> str:
    if not chunk.segments:
        return chunk.text.strip()
    overlap_start = 0 if chunk.index == 0 else AUDIO_OVERLAP_MILLISECONDS
    parts = [
        segment.text
        for segment in chunk.segments
        if segment.start_milliseconds >= overlap_start
    ]
    if not parts:
        return ""
    return "\n".join(parts).strip()


def _drop_overlap_segments(
    transcript: AudioTranscript,
    chunk_index: int,
) -> list[AudioTranscriptSegment]:
    overlap_start = 0 if chunk_index == 0 else AUDIO_OVERLAP_MILLISECONDS
    return [
        segment
        for segment in transcript.segments
        if segment.start_milliseconds >= overlap_start
    ]


def _transcribe_with_retry(
    transcriber: AudioTranscriptionClient,
    chunk_path: Path,
    prompt: str | None,
    source: str,
    chunk_index: int,
) -> AudioTranscript:
    last_error: Exception | None = None
    for attempt in range(AUDIO_TRANSCRIPTION_MAX_ATTEMPTS):
        try:
            transcript = transcriber.transcribe_audio(chunk_path, prompt=prompt)
            return AudioTranscript(
                text=_normalize_transcript(transcript.text),
                segments=transcript.segments,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < AUDIO_TRANSCRIPTION_MAX_ATTEMPTS:
                delay = AUDIO_TRANSCRIPTION_BACKOFF_SECONDS[attempt]
                RUN_LOGGER.warning(
                    "Retrying audio transcription source=%s chunk=%s "
                    "attempt=%s error=%s",
                    source,
                    chunk_index,
                    attempt + 1,
                    exc,
                )
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def _format_timestamp(milliseconds: int) -> str:
    total_seconds = milliseconds // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _ffmpeg_time(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


def _normalize_transcript(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validate_chunk(
    chunk: AudioTranscriptChunk,
    expected_index: int,
    expected_start: int,
    expected_end: int,
    source: str,
) -> AudioTranscriptChunk:
    if (
        chunk.index != expected_index
        or chunk.start_milliseconds != expected_start
        or chunk.end_milliseconds != expected_end
    ):
        raise ValueError(
            f"Cached audio transcription for {source!r} chunk {expected_index} "
            "has unexpected boundaries."
        )
    text = _normalize_transcript(chunk.text)
    if not text:
        raise ValueError(
            f"Cached audio transcription for {source!r} chunk {expected_index} "
            "is empty."
        )
    return chunk.model_copy(update={"text": text})


def _same_conversion(
    current: _AudioConversionManifest,
    expected: _AudioConversionManifest,
) -> bool:
    fields = (
        "source_hash",
        "model",
        "format_revision",
        "chunk_milliseconds",
        "overlap_milliseconds",
        "sample_rate",
        "bitrate",
        "duration_milliseconds",
        "chunk_count",
    )
    return all(getattr(current, field) == getattr(expected, field) for field in fields)


def _same_source_and_settings(
    manifest: _AudioConversionManifest,
    source_hash: str,
    model: str,
) -> bool:
    return (
        manifest.source_hash == source_hash
        and manifest.model == model
        and manifest.format_revision == AUDIO_FORMAT_REVISION
        and manifest.chunk_milliseconds == AUDIO_CHUNK_MILLISECONDS
        and manifest.overlap_milliseconds == AUDIO_OVERLAP_MILLISECONDS
        and manifest.sample_rate == AUDIO_SAMPLE_RATE
        and manifest.bitrate == AUDIO_BITRATE
    )


def _load_manifest(path: Path) -> _AudioConversionManifest | None:
    if not path.exists():
        return None
    return _AudioConversionManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    _write_text_atomic(path, model.model_dump_json(indent=2))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
