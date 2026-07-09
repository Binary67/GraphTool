from unittest.mock import Mock

import pytest

from graphtool.llm.config import ConfigError, load_azure_openai_config


def test_loads_azure_openai_config(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    endpoint = "https://example.openai.azure.com/openai/v1/"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.delenv(
        "GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY",
        raising=False,
    )

    config = load_azure_openai_config()

    load_dotenv.assert_called_once_with(override=True)
    assert config.endpoint == endpoint
    assert config.api_key == "test-key"
    assert config.model == "test-deployment"
    assert config.embedding_model == "embedding-deployment"
    assert config.embedding_batch_size == 4
    assert config.entity_resolution_min_candidate_similarity == 0.80


def test_missing_azure_openai_config_raises_clear_error(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.delenv(
        "GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY",
        raising=False,
    )

    with pytest.raises(ConfigError) as exc_info:
        load_azure_openai_config()

    load_dotenv.assert_called_once_with(override=True)
    message = str(exc_info.value)
    assert "AZURE_OPENAI_ENDPOINT" in message
    assert "AZURE_OPENAI_API_KEY" in message
    assert "AZURE_OPENAI_MODEL" in message
    assert "AZURE_OPENAI_EMBEDDING_MODEL" in message


def test_loads_custom_embedding_batch_size(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_BATCH_SIZE", "8")

    config = load_azure_openai_config()

    assert config.embedding_batch_size == 8


def test_blank_embedding_batch_size_uses_default(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_BATCH_SIZE", " ")

    config = load_azure_openai_config()

    assert config.embedding_batch_size == 4


def test_loads_custom_entity_resolution_min_candidate_similarity(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.setenv("GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY", "0.85")

    config = load_azure_openai_config()

    assert config.entity_resolution_min_candidate_similarity == 0.85


def test_blank_entity_resolution_min_candidate_similarity_uses_default(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.setenv("GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY", " ")

    config = load_azure_openai_config()

    assert config.entity_resolution_min_candidate_similarity == 0.80


@pytest.mark.parametrize("value", ["abc", "-0.1", "1.1", "nan"])
def test_invalid_entity_resolution_min_candidate_similarity_raises_clear_error(
    monkeypatch,
    value,
):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.setenv("GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY", value)

    with pytest.raises(ConfigError) as exc_info:
        load_azure_openai_config()

    assert "GRAPHTOOL_ENTITY_RESOLUTION_MIN_CANDIDATE_SIMILARITY" in str(exc_info.value)


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_embedding_batch_size_raises_clear_error(monkeypatch, value):
    load_dotenv = Mock()
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", load_dotenv)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_MODEL", "embedding-deployment")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_BATCH_SIZE", value)

    with pytest.raises(ConfigError) as exc_info:
        load_azure_openai_config()

    assert "AZURE_OPENAI_EMBEDDING_BATCH_SIZE" in str(exc_info.value)
