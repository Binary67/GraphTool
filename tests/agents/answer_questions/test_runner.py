from langchain_core.messages import AIMessage, ToolMessage

from graphtool.agents.answer_questions.runner import (
    MAX_AGENT_ITERATIONS,
    answer_question,
)
from graphtool.agents.answer_questions.types import (
    ChunkNeighborhood,
    ChunkReference,
    NeighborhoodChunk,
    RetrievedContext,
)
from graphtool.llm.config import AzureOpenAIConfig


class FakeGraph:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, payload, config):
        self.calls.append((payload, config))
        first = RetrievedContext(
            query="GraphTool agents",
            sources=["docs/agents.md"],
            chunk_references=[
                ChunkReference(
                    chunk_id="agents-chunk-0001",
                    source="docs/agents.md",
                    index=1,
                    heading_path=["Agents"],
                )
            ],
            context_text="Agent context",
        )
        neighborhood = ChunkNeighborhood(
            source="docs/agents.md",
            chunk_id="agents-chunk-0001",
            previous=None,
            current=NeighborhoodChunk(
                chunk_id="agents-chunk-0001",
                source="docs/agents.md",
                index=1,
                heading_path=["Agents"],
                text="Agent context",
            ),
            next=None,
        )
        second = RetrievedContext(
            query="GraphTool retrieval",
            sources=["docs/retrieval.md", "docs/agents.md"],
            chunk_references=[],
            context_text="Retrieval context",
        )
        return {
            "messages": [
                ToolMessage(content=first.model_dump_json(), tool_call_id="call-1"),
                ToolMessage(
                    content=neighborhood.model_dump_json(),
                    tool_call_id="call-2",
                ),
                ToolMessage(content=second.model_dump_json(), tool_call_id="call-3"),
                AIMessage(
                    content=(
                        "GraphTool can answer with an agent using retrieval "
                        "[docs/agents.md]."
                    )
                ),
            ]
        }


class FakeRuntime:
    graph_store = object()
    chunk_store = object()
    knowledge_base_store = object()
    fast_llm = object()
    chunk_embedding_store = object()


def _config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        flagship_deployment="flagship-deployment",
        fast_deployment="fast-deployment",
        embedding_deployment="embedding-deployment",
    )


def test_answer_question_returns_answer_and_retrieval_trace(monkeypatch):
    fake_graph = FakeGraph()
    fake_runtime = FakeRuntime()
    fake_model = object()
    fake_tool = object()
    fake_neighborhood_tool = object()
    runtime_calls = []
    tool_calls = []
    graph_calls = []
    monkeypatch.setattr(
        "graphtool.agents.answer_questions.runner.make_answer_chat_model",
        lambda config: fake_model,
    )

    def fake_create_runtime(config):
        runtime_calls.append(config)
        return fake_runtime

    def fake_make_tool(*args, **kwargs):
        tool_calls.append((args, kwargs))
        return fake_tool

    monkeypatch.setattr(
        "graphtool.agents.answer_questions.runner.create_runtime",
        fake_create_runtime,
    )
    monkeypatch.setattr(
        "graphtool.agents.answer_questions.runner.make_retrieve_knowledge_context_tool",
        fake_make_tool,
    )
    monkeypatch.setattr(
        "graphtool.agents.answer_questions.runner.make_get_chunk_neighborhood_tool",
        lambda chunk_store: fake_neighborhood_tool,
    )

    def fake_build_graph(model, tools):
        graph_calls.append((model, tools))
        return fake_graph

    monkeypatch.setattr(
        "graphtool.agents.answer_questions.runner.build_answer_question_graph",
        fake_build_graph,
    )
    config = _config()

    result = answer_question(
        "What can GraphTool answer?",
        config,
    )

    assert result.question == "What can GraphTool answer?"
    assert result.answer == (
        "GraphTool can answer with an agent using retrieval [docs/agents.md]."
    )
    assert result.sources == ["docs/agents.md", "docs/retrieval.md"]
    assert [retrieval.query for retrieval in result.retrievals] == [
        "GraphTool agents",
        "GraphTool retrieval",
    ]
    assert runtime_calls == [config]
    assert tool_calls == [
        (
            (fake_runtime.graph_store, fake_runtime.chunk_store),
            {
                "knowledge_base_store": fake_runtime.knowledge_base_store,
                "embedding_client": fake_runtime.fast_llm,
                "chunk_embedding_store": fake_runtime.chunk_embedding_store,
            },
        )
    ]
    assert graph_calls == [
        (fake_model, [fake_tool, fake_neighborhood_tool])
    ]
    assert fake_graph.calls == [
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What can GraphTool answer?",
                    }
                ]
            },
            {"recursion_limit": MAX_AGENT_ITERATIONS},
        )
    ]
