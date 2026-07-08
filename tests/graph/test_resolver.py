from typing import TypeVar, cast

from graphtool.graph.embedding_store import JsonEmbeddingStore
from graphtool.graph.resolver import EntityResolutionDecision, SemanticEntityResolver
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage

T = TypeVar("T")


class FakeEmbeddingClient:
    embedding_model = "fake-embedding-model"

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        label_line = text.splitlines()[0] if text else ""
        for key, vector in self.vectors.items():
            if key in label_line:
                return vector
        return [0.0, 1.0]


class FakeLLM:
    def __init__(self, decisions: list[EntityResolutionDecision] | None = None) -> None:
        self.decisions = decisions or []
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return cast(T, self.decisions[len(self.calls) - 1])


def test_resolver_merges_exact_id_and_preserves_aliases():
    resolver = SemanticEntityResolver(FakeLLM(), FakeEmbeddingClient())

    graph = resolver.combine(
        [
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="openai",
                        label="OpenAI",
                        type="Organization",
                        chunk_ids=["chunk-1"],
                    )
                ],
                edges=[],
            ),
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="openai",
                        label="OpenAI organization",
                        type="Organization",
                        aliases=["OpenAI Inc."],
                        chunk_ids=["chunk-2"],
                    )
                ],
                edges=[],
            ),
        ]
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "openai"
    assert graph.nodes[0].aliases == ["OpenAI organization", "OpenAI Inc."]
    assert graph.nodes[0].chunk_ids == ["chunk-1", "chunk-2"]


def test_resolver_merges_normalized_alias_match_without_llm():
    llm = FakeLLM()
    embedding = FakeEmbeddingClient()
    resolver = SemanticEntityResolver(llm, embedding)

    graph = resolver.combine(
        [
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="openai",
                        label="OpenAI",
                        type="Organization",
                        aliases=["OpenAI organization"],
                    )
                ],
                edges=[],
            ),
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="openai-org",
                        label="openai organization",
                        type="Organization",
                    )
                ],
                edges=[],
            ),
        ]
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "openai"
    assert llm.calls == []
    assert embedding.calls == []


def test_resolver_embeds_only_after_canonical_candidates_exist():
    embedding = FakeEmbeddingClient(
        {
            "First": [1.0, 0.0],
            "Second": [0.0, 1.0],
        }
    )
    resolver = SemanticEntityResolver(FakeLLM(), embedding)

    resolver.combine(
        [
            KnowledgeGraph(
                nodes=[
                    Node(id="first", label="First", type="Concept"),
                    Node(id="second", label="Second", type="Concept"),
                ],
                edges=[],
            )
        ]
    )

    assert "label: Second" in embedding.calls[0]
    assert any("label: First" in call for call in embedding.calls[1:])


def test_resolver_uses_embeddings_and_llm_to_merge_and_remap_edges():
    llm = FakeLLM(
        [
            EntityResolutionDecision(
                decision="merge",
                target_node_id="openai",
                confidence=0.95,
                aliases_to_add=["OpenAI org"],
            )
        ]
    )
    embedding = FakeEmbeddingClient(
        {
            "OpenAI organization": [1.0, 0.0],
            "OpenAI": [1.0, 0.0],
            "ChatGPT": [0.0, 1.0],
        }
    )
    resolver = SemanticEntityResolver(llm, embedding)

    graph = resolver.combine(
        [
            KnowledgeGraph(
                nodes=[
                    Node(id="openai", label="OpenAI", type="Organization"),
                    Node(id="chatgpt", label="ChatGPT", type="Product"),
                ],
                edges=[
                    Edge(
                        id="edge-a",
                        source="openai",
                        target="chatgpt",
                        label="develops",
                        chunk_ids=["chunk-1"],
                    )
                ],
            ),
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="openai-organization",
                        label="OpenAI organization",
                        type="Organization",
                    ),
                    Node(id="chatgpt", label="ChatGPT", type="Product"),
                ],
                edges=[
                    Edge(
                        id="edge-b",
                        source="openai-organization",
                        target="chatgpt",
                        label="develops",
                        chunk_ids=["chunk-2"],
                    )
                ],
            ),
        ]
    )

    assert {node.id for node in graph.nodes} == {"openai", "chatgpt"}
    openai = next(node for node in graph.nodes if node.id == "openai")
    assert openai.aliases == ["OpenAI organization", "OpenAI org"]
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "openai"
    assert graph.edges[0].target == "chatgpt"
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["chunk-1", "chunk-2"]


def test_resolver_keeps_related_entities_separate_when_llm_rejects_merge():
    llm = FakeLLM(
        [
            EntityResolutionDecision(
                decision="new",
                confidence=0.4,
            )
        ]
    )
    embedding = FakeEmbeddingClient(
        {
            "OpenAI API": [1.0, 0.0],
            "OpenAI": [1.0, 0.0],
        }
    )
    resolver = SemanticEntityResolver(llm, embedding)

    graph = resolver.combine(
        [
            KnowledgeGraph(
                nodes=[Node(id="openai", label="OpenAI", type="Organization")],
                edges=[],
            ),
            KnowledgeGraph(
                nodes=[Node(id="openai-api", label="OpenAI API", type="Service")],
                edges=[],
            ),
        ]
    )

    assert {node.id for node in graph.nodes} == {"openai", "openai-api"}


def test_resolver_reuses_matching_cached_embeddings(tmp_path):
    embedding = FakeEmbeddingClient({"OpenAI": [1.0, 0.0]})
    store = JsonEmbeddingStore(tmp_path / "embeddings.json")
    resolver = SemanticEntityResolver(FakeLLM(), embedding, store)

    resolver.combine(
        [
            KnowledgeGraph(
                nodes=[Node(id="openai", label="OpenAI", type="Organization")],
                edges=[],
            )
        ]
    )
    first_call_count = len(embedding.calls)

    resolver = SemanticEntityResolver(FakeLLM(), embedding, store)
    resolver.combine(
        [
            KnowledgeGraph(
                nodes=[Node(id="openai", label="OpenAI", type="Organization")],
                edges=[],
            )
        ]
    )

    assert len(embedding.calls) == first_call_count
    assert store.load()["openai"].vector == [1.0, 0.0]
