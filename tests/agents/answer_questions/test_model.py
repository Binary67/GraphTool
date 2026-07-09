from graphtool.agents.answer_questions.model import make_answer_chat_model
from graphtool.llm.config import AzureOpenAIConfig


def test_make_answer_chat_model_uses_flagship_azure_config():
    config = AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        flagship_deployment="flagship-deployment",
        fast_deployment="fast-deployment",
        embedding_deployment="embedding-deployment",
    )

    model = make_answer_chat_model(config)

    assert model.model_name == "flagship-deployment"
    assert model.openai_api_base == "https://example.openai.azure.com/openai/v1/"
    assert model.openai_api_key.get_secret_value() == "test-key"
