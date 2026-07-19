import base64
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import OpenAI

from graphtool.llm.config import AzureOpenAIConfig
from graphtool.llm.types import (
    LLMImageContent,
    LLMMessage,
    LLMTextContent,
    LLMTextResponse,
)

T = TypeVar("T")


class AzureOpenAIClient:
    def __init__(self, config: AzureOpenAIConfig, *, text_deployment: str) -> None:
        self._config = config
        self._text_deployment = text_deployment
        self._client = OpenAI(base_url=config.endpoint, api_key=config.api_key)

    @property
    def embedding_model(self) -> str:
        return self._config.embedding_deployment

    @property
    def text_model(self) -> str:
        return self._text_deployment

    def generate_text(self, messages: Sequence[LLMMessage]) -> LLMTextResponse:
        response = self._client.responses.create(
            model=self._text_deployment,
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
            model=self._text_deployment,
            input=_to_response_input(messages),
            text_format=response_model,
        )

        return response.output_parsed

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for batch in _batches(texts, self._config.embedding_batch_size):
            response = self._client.embeddings.create(
                model=self._config.embedding_deployment,
                input=batch,
            )
            vectors.extend(list(item.embedding) for item in response.data)
        return vectors


def create_azure_openai_agent_model(
    config: AzureOpenAIConfig,
) -> BaseChatModel:
    return ChatOpenAI(
        model=config.agent_deployment,
        base_url=config.endpoint,
        api_key=config.api_key,
    )


def _to_response_input(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "content": _to_response_content(message),
        }
        for message in messages
    ]


def _to_response_content(message: LLMMessage) -> str | list[dict[str, str]]:
    if isinstance(message.content, str):
        return message.content

    content = []
    for part in message.content:
        if isinstance(part, LLMTextContent):
            content.append({"type": "input_text", "text": part.text})
            continue

        if isinstance(part, LLMImageContent):
            encoded = base64.b64encode(part.data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{part.media_type};base64,{encoded}",
                    "detail": part.detail,
                }
            )
            continue

        raise TypeError(f"Unsupported LLM content part: {type(part).__name__}")
    return content


def _batches(values: Sequence[str], batch_size: int) -> list[list[str]]:
    return [
        list(values[index : index + batch_size])
        for index in range(0, len(values), batch_size)
    ]
