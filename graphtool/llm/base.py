from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from graphtool.llm.types import LLMMessage, LLMTextResponse

T = TypeVar("T")


class LLMClient(Protocol):
    """Common interface implemented by all LLM providers."""

    @property
    def text_model(self) -> str:
        """Return the text-generation model identifier."""
        ...

    def generate_text(self, messages: Sequence[LLMMessage]) -> LLMTextResponse:
        """Generate plain text from a sequence of messages."""
        ...

    def generate_structured(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
    ) -> T:
        """Generate a structured response parsed into response_model."""
        ...


class EmbeddingClient(Protocol):
    """Common interface implemented by embedding providers."""

    @property
    def embedding_model(self) -> str:
        """Return the embedding model identifier."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple text values."""
        ...


class AudioTranscriptionClient(Protocol):
    """Interface for file-based audio transcription providers."""

    @property
    def transcription_model(self) -> str:
        """Return the transcription model or deployment identifier."""
        ...

    def transcribe_audio(self, path: str | Path, *, prompt: str | None = None) -> str:
        """Transcribe an audio file without summarizing it."""
        ...
