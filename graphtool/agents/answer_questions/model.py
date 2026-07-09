from langchain_openai import ChatOpenAI

from graphtool.llm.config import AzureOpenAIConfig


def make_answer_chat_model(config: AzureOpenAIConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.flagship_deployment,
        base_url=config.endpoint,
        api_key=config.api_key,
    )
