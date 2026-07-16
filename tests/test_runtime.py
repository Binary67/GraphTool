import pytest

from graphtool.chunking.types import Chunk
from graphtool.graph.types import KnowledgeGraph, Node
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.runtime import create_runtime, default_paths


class FakeAzureOpenAIClient:
    instances = []
    embedding_model = "embedding-deployment"

    def __init__(self, config, *, text_deployment):
        self.config = config
        self.text_deployment = text_deployment
        self.embedding_calls = []
        FakeAzureOpenAIClient.instances.append(self)

    def embed_texts(self, texts):
        batch = list(texts)
        self.embedding_calls.extend(batch)
        return [
            [1.0, 0.0]
            if "install hangs" in text or "Setup stalls" in text
            else [0.0, 1.0]
            for text in batch
        ]


def _config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        fast_deployment="fast-deployment",
        embedding_deployment="embedding-deployment",
    )


def _runtime(monkeypatch, tmp_path):
    FakeAzureOpenAIClient.instances = []
    monkeypatch.setattr("graphtool.runtime.AzureOpenAIClient", FakeAzureOpenAIClient)
    return create_runtime(_config(), paths=default_paths(tmp_path))


def test_default_paths_match_project_layout(tmp_path):
    paths = default_paths(tmp_path)

    assert paths.root == tmp_path
    assert paths.documents_dir == tmp_path / "documents"
    assert paths.chunks_dir == tmp_path / "data" / "chunks"
    assert paths.chunk_extractions_dir == tmp_path / "data" / "chunk_extractions"
    assert paths.graphs_dir == tmp_path / "data" / "graphs"
    assert paths.graph_embeddings_dir == tmp_path / "data" / "graph_embeddings"
    assert paths.chunk_embeddings_path == tmp_path / "data" / "chunk_embeddings.json"
    assert paths.knowledge_base_path == tmp_path / "data" / "knowledge_base.json"
    assert paths.knowledge_base_embeddings_path == (
        tmp_path / "data" / "knowledge_base_embeddings.json"
    )
    assert paths.taxonomy_suggestions_path == (
        tmp_path / "data" / "taxonomy_suggestions.json"
    )
    assert paths.dropped_edges_path == tmp_path / "data" / "dropped_edges.jsonl"
    assert paths.logs_dir == tmp_path / "logs"
    assert paths.visualizations_dir == tmp_path / "data" / "visualizations"


def test_create_runtime_uses_fast_deployment_for_runtime_llm(monkeypatch, tmp_path):
    FakeAzureOpenAIClient.instances = []
    monkeypatch.setattr("graphtool.runtime.AzureOpenAIClient", FakeAzureOpenAIClient)
    config = _config()
    paths = default_paths(tmp_path)

    runtime = create_runtime(config, paths=paths)

    assert runtime.paths == paths
    assert runtime.fast_llm is FakeAzureOpenAIClient.instances[0]
    assert runtime.fast_llm.config is config
    assert runtime.fast_llm.text_deployment == "fast-deployment"
    assert runtime.chunk_extraction_store.load("docs/missing.md") == {}


def test_search_uses_combined_graph_all_chunks_and_top_chunks(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    chunks = [
        Chunk(
            id="pydantic-chunk-0000",
            source="docs/pydantic.md",
            index=0,
            text="Pydantic handles data validation.",
        ),
        Chunk(
            id="fastapi-chunk-0000",
            source="docs/fastapi.md",
            index=0,
            text="FastAPI handles request validation.",
        ),
    ]
    runtime.chunk_store.save(chunks[0].source, [chunks[0]])
    runtime.chunk_store.save(chunks[1].source, [chunks[1]])
    runtime.knowledge_base_store.save(
        KnowledgeGraph(
            nodes=[
                Node(
                    id="pydantic",
                    label="Pydantic",
                    type="Library",
                    chunk_ids=[chunks[0].id],
                ),
                Node(
                    id="fastapi",
                    label="FastAPI",
                    type="Framework",
                    chunk_ids=[chunks[1].id],
                ),
            ],
            edges=[],
        )
    )

    result = runtime.search("validation", top_chunks=2)
    limited_result = runtime.search("validation", top_chunks=1)

    assert {hit.chunk.id for hit in result.chunks} == {
        "pydantic-chunk-0000",
        "fastapi-chunk-0000",
    }
    assert {node.id for hit in result.chunks for node in hit.linked_nodes} == {
        "pydantic",
        "fastapi",
    }
    assert len(limited_result.chunks) == 1


def test_search_uses_runtime_embeddings_and_cache(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    chunks = [
        Chunk(
            id="deploy-chunk-0000",
            source="docs/deploy.md",
            index=0,
            text="Setup stalls after authentication.",
        ),
        Chunk(
            id="billing-chunk-0001",
            source="docs/deploy.md",
            index=1,
            text="Billing exports finish normally.",
        ),
    ]
    runtime.chunk_store.save("docs/deploy.md", chunks)
    runtime.knowledge_base_store.save(KnowledgeGraph(nodes=[], edges=[]))

    result = runtime.search("install hangs", top_chunks=1)

    assert [hit.chunk.id for hit in result.chunks] == ["deploy-chunk-0000"]
    assert runtime.fast_llm.embedding_calls
    assert runtime.chunk_embedding_store.exists() is True


def test_search_requires_synchronized_knowledge_base(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Synchronize documents before searching",
    ):
        runtime.search("validation")
