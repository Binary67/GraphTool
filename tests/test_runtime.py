from graphtool.llm.config import AzureOpenAIConfig
from graphtool.runtime import create_runtime, default_paths


class FakeAzureOpenAIClient:
    instances = []

    def __init__(self, config, *, text_deployment):
        self.config = config
        self.text_deployment = text_deployment
        FakeAzureOpenAIClient.instances.append(self)


def _config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        fast_deployment="fast-deployment",
        embedding_deployment="embedding-deployment",
    )


def test_default_paths_match_project_layout(tmp_path):
    paths = default_paths(tmp_path)

    assert paths.root == tmp_path
    assert paths.documents_dir == tmp_path / "documents"
    assert paths.chunks_dir == tmp_path / "data" / "chunks"
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
