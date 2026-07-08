"""Provider-neutral LLM interface and Azure OpenAI implementation."""

from graphtool.llm.azure_openai import AzureOpenAIClient
from graphtool.llm.base import EmbeddingClient, LLMClient
from graphtool.llm.config import AzureOpenAIConfig, ConfigError, load_azure_openai_config
from graphtool.llm.types import LLMMessage, LLMTextResponse

__all__ = [
    "AzureOpenAIClient",
    "AzureOpenAIConfig",
    "ConfigError",
    "EmbeddingClient",
    "LLMClient",
    "LLMMessage",
    "LLMTextResponse",
    "load_azure_openai_config",
]
