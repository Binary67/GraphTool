from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class LLMTextContent:
    text: str


@dataclass(frozen=True)
class LLMImageContent:
    data: bytes
    media_type: Literal["image/png"] = "image/png"
    detail: Literal["low", "high", "auto"] = "high"


LLMContentPart: TypeAlias = LLMTextContent | LLMImageContent


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str | tuple[LLMContentPart, ...]


@dataclass(frozen=True)
class LLMTextResponse:
    content: str
    response_id: str | None = None
    model: str | None = None
