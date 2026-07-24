"""Provider-neutral LLM interface and Azure OpenAI implementation."""

from graphtool.llm.azure_openai import (
    AGENT_MAX_RETRIES,
    AGENT_REQUEST_TIMEOUT_SECONDS,
    AzureOpenAIAudioTranscriber,
    AzureOpenAIClient,
    create_azure_openai_agent_model,
    create_azure_openai_fast_agent_model,
)
from graphtool.llm.base import AudioTranscriptionClient, EmbeddingClient, LLMClient
from graphtool.llm.config import AzureOpenAIConfig, ConfigError, load_azure_openai_config
from graphtool.llm.types import (
    AudioTranscript,
    AudioTranscriptSegment,
    LLMContentPart,
    LLMImageContent,
    LLMMessage,
    LLMTextContent,
    LLMTextResponse,
)

__all__ = [
    "AudioTranscriptionClient",
    "AudioTranscript",
    "AudioTranscriptSegment",
    "AGENT_MAX_RETRIES",
    "AGENT_REQUEST_TIMEOUT_SECONDS",
    "AzureOpenAIAudioTranscriber",
    "AzureOpenAIClient",
    "AzureOpenAIConfig",
    "ConfigError",
    "EmbeddingClient",
    "LLMContentPart",
    "LLMClient",
    "LLMImageContent",
    "LLMMessage",
    "LLMTextContent",
    "LLMTextResponse",
    "create_azure_openai_agent_model",
    "create_azure_openai_fast_agent_model",
    "load_azure_openai_config",
]
