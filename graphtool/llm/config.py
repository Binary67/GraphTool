import os
from dataclasses import dataclass
from typing import cast

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when required LLM configuration is missing."""


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    model: str


def load_azure_openai_config() -> AzureOpenAIConfig:
    load_dotenv(override=True)

    names = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_MODEL",
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
    )
