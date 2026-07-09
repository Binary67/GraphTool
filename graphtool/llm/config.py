import math
import os
from dataclasses import dataclass
from typing import cast

from dotenv import load_dotenv

DEFAULT_EMBEDDING_BATCH_SIZE = 4
DEFAULT_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY = 0.80
FLAGSHIP_DEPLOYMENT_ENV = "AZURE_OPENAI_FLAGSHIP_DEPLOYMENT"
FAST_DEPLOYMENT_ENV = "AZURE_OPENAI_FAST_DEPLOYMENT"
EMBEDDING_DEPLOYMENT_ENV = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
EMBEDDING_BATCH_SIZE_ENV = "AZURE_OPENAI_EMBEDDING_BATCH_SIZE"
ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY_ENV = (
    "GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY"
)


class ConfigError(ValueError):
    """Raised when required LLM configuration is missing."""


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    flagship_deployment: str
    fast_deployment: str
    embedding_deployment: str
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    entity_resolution_min_candidate_similarity: float = (
        DEFAULT_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY
    )


def load_azure_openai_config() -> AzureOpenAIConfig:
    load_dotenv(override=True)

    names = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        FLAGSHIP_DEPLOYMENT_ENV,
        FAST_DEPLOYMENT_ENV,
        EMBEDDING_DEPLOYMENT_ENV,
    ]
    values = {name: os.getenv(name) for name in names}
    missing = [name for name, value in values.items() if not value]

    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required Azure OpenAI environment variables: {joined}")

    return AzureOpenAIConfig(
        endpoint=cast(str, values["AZURE_OPENAI_ENDPOINT"]),
        api_key=cast(str, values["AZURE_OPENAI_API_KEY"]),
        flagship_deployment=cast(str, values[FLAGSHIP_DEPLOYMENT_ENV]),
        fast_deployment=cast(str, values[FAST_DEPLOYMENT_ENV]),
        embedding_deployment=cast(str, values[EMBEDDING_DEPLOYMENT_ENV]),
        embedding_batch_size=_embedding_batch_size(os.getenv(EMBEDDING_BATCH_SIZE_ENV)),
        entity_resolution_min_candidate_similarity=(
            _entity_resolution_min_candidate_similarity(
                os.getenv(ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY_ENV)
            )
        ),
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


def _entity_resolution_min_candidate_similarity(value: str | None) -> float:
    if value is None or value.strip() == "":
        return DEFAULT_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY

    try:
        similarity = float(value)
    except ValueError as exc:
        raise ConfigError(
            f"{ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY_ENV} must be between 0.0 and 1.0"
        ) from exc

    if not math.isfinite(similarity) or similarity < 0.0 or similarity > 1.0:
        raise ConfigError(
            f"{ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY_ENV} must be between 0.0 and 1.0"
        )

    return similarity
