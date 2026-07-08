import pytest

from graphtool.llm.config import ConfigError, load_azure_openai_config


def test_loads_azure_openai_config(monkeypatch):
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", lambda: None)
    endpoint = "https://example.openai.azure.com/openai/v1/"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "test-deployment")

    config = load_azure_openai_config()

    assert config.endpoint == endpoint
    assert config.api_key == "test-key"
    assert config.model == "test-deployment"


def test_missing_azure_openai_config_raises_clear_error(monkeypatch):
    monkeypatch.setattr("graphtool.llm.config.load_dotenv", lambda: None)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)

    with pytest.raises(ConfigError) as exc_info:
        load_azure_openai_config()

    message = str(exc_info.value)
    assert "AZURE_OPENAI_ENDPOINT" in message
    assert "AZURE_OPENAI_API_KEY" in message
    assert "AZURE_OPENAI_MODEL" in message
