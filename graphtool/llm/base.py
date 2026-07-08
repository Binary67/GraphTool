from collections.abc import Sequence
from typing import Protocol, TypeVar

from graphtool.llm.types import LLMMessage, LLMTextResponse

T = TypeVar("T")


class LLMClient(Protocol):
    """Common interface implemented by all LLM providers."""

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
