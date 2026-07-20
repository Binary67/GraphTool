from pydantic import BaseModel

from graphtool.llm.azure_openai import (
    AzureOpenAIAudioTranscriber,
    AzureOpenAIClient,
    create_azure_openai_agent_model,
)
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.llm.types import LLMImageContent, LLMMessage, LLMTextContent


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
        self.audio = FakeAudio()
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


class FakeAudio:
    def __init__(self):
        self.transcriptions = FakeTranscriptions()


class FakeTranscriptions:
    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return type("Transcription", (), {"text": "transcribed audio"})()


class Person(BaseModel):
    name: str


def _config(**overrides):
    values = {
        "endpoint": "https://example.openai.azure.com/openai/v1/",
        "api_key": "test-key",
        "agent_deployment": "agent-deployment",
        "fast_deployment": "fast-deployment",
        "embedding_deployment": "embedding-deployment",
        "transcription_deployment": "transcription-deployment",
    }
    values.update(overrides)
    return AzureOpenAIConfig(**values)


def test_constructs_openai_client_with_exact_config(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config()

    client = AzureOpenAIClient(config, text_deployment=config.fast_deployment)

    assert len(FakeOpenAI.instances) == 1
    assert FakeOpenAI.instances[0].base_url == config.endpoint
    assert FakeOpenAI.instances[0].api_key == config.api_key
    assert client.text_model == config.fast_deployment


def test_constructs_agent_model_with_dedicated_deployment(monkeypatch):
    calls = []

    def fake_chat_openai(**kwargs):
        calls.append(kwargs)
        return "agent-model"

    monkeypatch.setattr("graphtool.llm.azure_openai.ChatOpenAI", fake_chat_openai)
    config = _config()

    model = create_azure_openai_agent_model(config)

    assert model == "agent-model"
    assert calls == [
        {
            "model": "agent-deployment",
            "base_url": config.endpoint,
            "api_key": config.api_key,
        }
    ]


def test_transcribes_audio_with_dedicated_deployment(monkeypatch, tmp_path):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    path = tmp_path / "chunk.mp3"
    path.write_bytes(b"audio")

    transcriber = AzureOpenAIAudioTranscriber(_config())
    text = transcriber.transcribe_audio(path, prompt="Previous context")

    assert transcriber.transcription_model == "transcription-deployment"
    assert text == "transcribed audio"
    call = FakeOpenAI.instances[0].audio.transcriptions.create_calls[0]
    assert call["model"] == "transcription-deployment"
    assert call["prompt"] == "Previous context"
    assert call["response_format"] == "json"
    assert call["file"].name == str(path)
    assert call["file"].closed is True


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


def test_generate_structured_serializes_multimodal_content(monkeypatch):
    FakeOpenAI.instances = []
    monkeypatch.setattr("graphtool.llm.azure_openai.OpenAI", FakeOpenAI)
    config = _config()
    client = AzureOpenAIClient(config, text_deployment=config.fast_deployment)

    client.generate_structured(
        [
            LLMMessage(
                role="user",
                content=(
                    LLMTextContent(text="Convert page 1."),
                    LLMImageContent(data=b"png-bytes"),
                ),
            )
        ],
        Person,
    )

    assert FakeOpenAI.instances[0].responses.parse_calls == [
        {
            "model": "fast-deployment",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Convert page 1."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,cG5nLWJ5dGVz",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text_format": Person,
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
