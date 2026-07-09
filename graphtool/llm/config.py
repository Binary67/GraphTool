import os
from dataclasses import dataclass
from typing import cast

from dotenv import load_dotenv

DEFAULT_EMBEDDING_BATCH_SIZE = 4
EMBEDDING_BATCH_SIZE_ENV = "AZURE_OPENAI_EMBEDDING_BATCH_SIZE"


class ConfigError(ValueError):
    """Raised when required LLM configuration is missing."""


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    model: str
    embedding_model: str
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE


def load_azure_openai_config() -> AzureOpenAIConfig:
    load_dotenv(override=True)

    names = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_MODEL",
        "AZURE_OPENAI_EMBEDDING_MODEL",
    ]
    values = {name: os.getenv(name) for name in names}
    missing = [name for name, value in values.items() if not value]

    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required Azure OpenAI environment variables: {joined}")

    return AzureOpenAIConfig(
        endpoint=cast(str, values["AZURE_OPENAI_ENDPOINT"]),
        api_key=cast(str, values["AZURE_OPENAI_API_KEY"]),
        model=cast(str, values["AZURE_OPENAI_MODEL"]),
        embedding_model=cast(str, values["AZURE_OPENAI_EMBEDDING_MODEL"]),
        embedding_batch_size=_embedding_batch_size(os.getenv(EMBEDDING_BATCH_SIZE_ENV)),
    )


def _embedding_batch_size(value: str | None) -> int:
    if value is None or value.strip() == "":
        return DEFAULT_EMBEDDING_BATCH_SIZE

    try:
        batch_size = int(value)
    except ValueError as exc:
        raise ConfigError(
            f"{EMBEDDING_BATCH_SIZE_ENV} must be a positive integer"
        ) from exc

    if batch_size < 1:
        raise ConfigError(f"{EMBEDDING_BATCH_SIZE_ENV} must be a positive integer")

    return batch_size
