from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMTextResponse:
    content: str
    response_id: str | None = None
    model: str | None = None
