from unittest.mock import Mock

import pytest

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, KnowledgeGraph, Node
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


class FakeAudioTranscriber:
    instances = []

    def __init__(self, config):
        self.config = config
        FakeAudioTranscriber.instances.append(self)


def _config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        agent_deployment="agent-deployment",
        fast_deployment="fast-deployment",
        embedding_deployment="embedding-deployment",
        transcription_deployment="transcription-deployment",
    )


def _runtime(monkeypatch, tmp_path):
    FakeAzureOpenAIClient.instances = []
    monkeypatch.setattr("graphtool.runtime.AzureOpenAIClient", FakeAzureOpenAIClient)
    monkeypatch.setattr(
        "graphtool.runtime.AzureOpenAIAudioTranscriber",
        FakeAudioTranscriber,
    )
    return create_runtime(_config(), paths=default_paths(tmp_path))


def test_default_paths_match_project_layout(tmp_path):
    paths = default_paths(tmp_path)

    assert paths.root == tmp_path
    assert paths.documents_dir == tmp_path / "documents"
    assert paths.audio_transcriptions_dir == tmp_path / "data" / "audio_transcriptions"
    assert paths.pdf_conversions_dir == tmp_path / "data" / "pdf_conversions"
    assert paths.presentation_conversions_dir == (
        tmp_path / "data" / "presentation_conversions"
    )
    assert paths.chunk_extractions_dir == tmp_path / "data" / "chunk_extractions"
    assert paths.graphs_dir == tmp_path / "data" / "graphs"
    assert paths.knowledge_base_path == tmp_path / "data" / "knowledge_base.json"
    assert paths.db_path == tmp_path / "data" / "graphtool.db"
    assert paths.dropped_edges_path == tmp_path / "data" / "dropped_edges.jsonl"
    assert paths.logs_dir == tmp_path / "logs"
    assert paths.visualizations_dir == tmp_path / "data" / "visualizations"


def test_create_runtime_uses_fast_deployment_for_runtime_llm(monkeypatch, tmp_path):
    FakeAzureOpenAIClient.instances = []
    monkeypatch.setattr("graphtool.runtime.AzureOpenAIClient", FakeAzureOpenAIClient)
    monkeypatch.setattr(
        "graphtool.runtime.AzureOpenAIAudioTranscriber",
        FakeAudioTranscriber,
    )
    config = _config()
    paths = default_paths(tmp_path)

    runtime = create_runtime(config, paths=paths)

    assert runtime.paths == paths
    assert runtime.fast_llm is FakeAzureOpenAIClient.instances[0]
    assert runtime.fast_llm.config is config
    assert runtime.fast_llm.text_deployment == "fast-deployment"
    assert runtime.audio_transcriber.config is config
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

    runtime.prepare_search()
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
    assert result.graph_paths == []
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

    runtime.prepare_search()
    result = runtime.search("install hangs", top_chunks=1)

    assert [hit.chunk.id for hit in result.chunks] == ["deploy-chunk-0000"]
    assert runtime.fast_llm.embedding_calls
    assert runtime.chunk_embedding_store.exists() is True


def test_prepared_search_reuses_loaded_corpus_and_embeds_each_query_once(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(monkeypatch, tmp_path)
    chunk = Chunk(
        id="deploy-chunk-0000",
        source="docs/deploy.md",
        index=0,
        text="Setup stalls after authentication.",
    )
    runtime.chunk_store.save(chunk.source, [chunk])
    runtime.knowledge_base_store.save(KnowledgeGraph(nodes=[], edges=[]))
    load_graph = Mock(wraps=runtime.knowledge_base_store.load)
    load_chunks = Mock(wraps=runtime.chunk_store.load_all)
    monkeypatch.setattr(runtime.knowledge_base_store, "load", load_graph)
    monkeypatch.setattr(runtime.chunk_store, "load_all", load_chunks)

    runtime.prepare_search()
    runtime.fast_llm.embedding_calls.clear()
    runtime.search("install hangs")
    runtime.search("deployment stalls")

    assert load_graph.call_count == 1
    assert load_chunks.call_count == 1
    assert runtime.fast_llm.embedding_calls == [
        "install hangs",
        "deployment stalls",
    ]


def test_prepare_search_refreshes_the_retrieval_snapshot(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    source = "docs/guide.md"
    runtime.chunk_store.save(
        source,
        [
            Chunk(
                id="guide-chunk-0000",
                source=source,
                index=0,
                text="Setup stalls after authentication.",
            )
        ],
    )
    runtime.knowledge_base_store.save(KnowledgeGraph(nodes=[], edges=[]))
    runtime.prepare_search()

    runtime.chunk_store.save(
        source,
        [
            Chunk(
                id="guide-chunk-0000",
                source=source,
                index=0,
                text="Billing exports finish normally.",
            )
        ],
    )

    assert runtime.search("billing exports").chunks == []

    runtime.prepare_search()
    refreshed = runtime.search("billing exports")

    assert [hit.chunk.text for hit in refreshed.chunks] == [
        "Billing exports finish normally."
    ]


def test_search_combines_direct_chunks_and_graph_paths(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    chunks = [
        Chunk(
            id="alpha-beta",
            source="docs/graph.md",
            index=0,
            text="Alpha uses Beta.",
        ),
        Chunk(
            id="beta-gamma",
            source="docs/graph.md",
            index=1,
            text="Beta depends on Gamma.",
        ),
    ]
    runtime.chunk_store.save("docs/graph.md", chunks)
    runtime.knowledge_base_store.save(
        KnowledgeGraph(
            nodes=[
                Node(
                    id="alpha",
                    label="Alpha",
                    type="System",
                    chunk_ids=["alpha-beta"],
                ),
                Node(
                    id="beta",
                    label="Beta",
                    type="Component",
                    chunk_ids=["alpha-beta", "beta-gamma"],
                ),
                Node(
                    id="gamma",
                    label="Gamma",
                    type="Service",
                    chunk_ids=["beta-gamma"],
                ),
            ],
            edges=[
                Edge(
                    id="alpha-beta-edge",
                    source="alpha",
                    target="beta",
                    label="uses",
                    chunk_ids=["alpha-beta"],
                ),
                Edge(
                    id="beta-gamma-edge",
                    source="beta",
                    target="gamma",
                    label="depends on",
                    chunk_ids=["beta-gamma"],
                ),
            ],
        )
    )

    runtime.prepare_search()
    result = runtime.search("How is Alpha related to Gamma?")

    assert any(
        [edge.id for edge in path.edges]
        == ["alpha-beta-edge", "beta-gamma-edge"]
        for path in result.graph_paths
    )
    assert {hit.chunk.id for hit in result.chunks} == {
        "alpha-beta",
        "beta-gamma",
    }


def test_search_requires_synchronized_knowledge_base(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Synchronize documents before searching",
    ):
        runtime.prepare_search()


def test_search_requires_preparation(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    runtime.knowledge_base_store.save(KnowledgeGraph(nodes=[], edges=[]))

    with pytest.raises(RuntimeError, match="Call prepare_search"):
        runtime.search("validation")
