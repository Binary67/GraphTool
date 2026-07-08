from collections.abc import Sequence
from typing import TypeVar

from openai import OpenAI

from graphtool.llm.config import AzureOpenAIConfig
from graphtool.llm.types import LLMMessage, LLMTextResponse

T = TypeVar("T")


class AzureOpenAIClient:
    def __init__(self, config: AzureOpenAIConfig) -> None:
        self._config = config
        self._client = OpenAI(base_url=config.endpoint, api_key=config.api_key)

    def generate_text(self, messages: Sequence[LLMMessage]) -> LLMTextResponse:
        response = self._client.responses.create(
            model=self._config.model,
            input=_to_response_input(messages),
        )

        return LLMTextResponse(
            content=response.output_text,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
        )

    def generate_structured(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
    ) -> T:
        response = self._client.responses.parse(
            model=self._config.model,
            input=_to_response_input(messages),
            text_format=response_model,
        )

        return response.output_parsed


def _to_response_input(messages: Sequence[LLMMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]
