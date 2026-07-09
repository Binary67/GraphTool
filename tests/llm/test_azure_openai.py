from pydantic import BaseModel

from graphtool.llm.azure_openai import AzureOpenAIClient
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.llm.types import LLMMessage


class FakeResponses:
    def __init__(self):
        self.create_calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeTextResponse()

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return FakeStructuredResponse()


class FakeOpenAI:
    instances = []

    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.responses = FakeResponses()
        self.embeddings = FakeEmbeddings()
        FakeOpenAI.instances.append(self)


class FakeTextResponse:
    output_text = "hello"
    id = "response-123"
    model = "text-deployment"


class FakeStructuredResponse:
    output_parsed = {"name": "Ada"}


class FakeEmbeddings:
    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        texts = kwargs["input"]
        if isinstance(texts, str):
            texts = [texts]
        return FakeEmbeddingResponse(texts)


class FakeEmbeddingResponse:
    def __init__(self, texts):
        self.data = [
            type(
                "Embedding",
                (),
                {"embedding": [float(index), float(len(text))]},
            )()
            for index, text in enumerate(texts)
        ]


class Person(BaseModel):
    name: str


def _config(**overrides):
    values = {
        "endpoint": "https://example.openai.azure.com/openai/v1/",
        "api_key": "test-key",
        "flagship_deployment": "flagship-deployment",
        "fast_deployment": "fast-deployment",
        "embedding_deployment": "embedding-deployment",
    }
    values.update(overrides)
    return AzureOpenAIConfig(**values)


def test_constructs_openai_client_with_exact_config(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config()

    AzureOpenAIClient(config, text_deployment=config.fast_deployment)

    assert len(FakeOpenAI.instances) == 1
    assert FakeOpenAI.instances[0].base_url == config.endpoint
    assert FakeOpenAI.instances[0].api_key == config.api_key


def test_generate_text_uses_responses_create(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config()
    client = AzureOpenAIClient(config, text_deployment="text-deployment")

    response = client.generate_text(
        [
            LLMMessage(role="system", content="You are concise."),
            LLMMessage(role="user", content="Say hello."),
        ]
    )

    assert response.content == "hello"
    assert response.response_id == "response-123"
    assert response.model == "text-deployment"
    assert FakeOpenAI.instances[0].responses.create_calls == [
        {
            "model": "text-deployment",
            "input": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hello."},
            ],
        }
    ]


def test_generate_structured_uses_responses_parse(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config()
    client = AzureOpenAIClient(config, text_deployment="text-deployment")

    parsed = client.generate_structured(
        [LLMMessage(role="user", content="Extract the person.")],
        Person,
    )

    assert parsed == {"name": "Ada"}
    assert FakeOpenAI.instances[0].responses.parse_calls == [
        {
            "model": "text-deployment",
            "input": [
                {"role": "user", "content": "Extract the person."},
            ],
            "text_format": Person,
        }
    ]


def test_embed_text_uses_embeddings_create(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config()
    client = AzureOpenAIClient(config, text_deployment="text-deployment")

    embedding = client.embed_text("OpenAI organization")

    assert client.embedding_model == "embedding-deployment"
    assert embedding == [0.0, 19.0]
    assert FakeOpenAI.instances[0].embeddings.create_calls == [
        {
            "model": "embedding-deployment",
            "input": ["OpenAI organization"],
        }
    ]


def test_embed_texts_batches_inputs_and_preserves_order(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config(embedding_batch_size=2)
    client = AzureOpenAIClient(config, text_deployment="text-deployment")

    embeddings = client.embed_texts(["alpha", "beta", "gamma"])

    assert embeddings == [[0.0, 5.0], [1.0, 4.0], [0.0, 5.0]]
    assert FakeOpenAI.instances[0].embeddings.create_calls == [
        {
            "model": "embedding-deployment",
            "input": ["alpha", "beta"],
        },
        {
            "model": "embedding-deployment",
            "input": ["gamma"],
        },
    ]
