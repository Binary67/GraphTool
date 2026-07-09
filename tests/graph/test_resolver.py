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
        self.batch_calls: list[list[str]] = []

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.batch_calls.append(batch)
        self.calls.extend(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
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


def test_resolver_skips_embedding_candidates_with_different_types():
    llm = FakeLLM()
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
    assert llm.calls == []
    assert embedding.calls == []
    assert embedding.batch_calls == []


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


def test_resolver_batches_uncached_candidate_embeddings():
    embedding = FakeEmbeddingClient(
        {
            "Alpha": [1.0, 0.0, 0.0],
            "Beta": [0.0, 1.0, 0.0],
            "Gamma": [0.0, 0.0, 1.0],
        }
    )
    resolver = SemanticEntityResolver(
        FakeLLM(),
        embedding,
        min_candidate_similarity=1.1,
    )
    existing = KnowledgeGraph(
        nodes=[
            Node(id="alpha", label="Alpha", type="Concept"),
            Node(id="beta", label="Beta", type="Concept"),
        ],
        edges=[],
    )

    graph = resolver.combine_into(
        existing,
        [
            KnowledgeGraph(
                nodes=[Node(id="gamma", label="Gamma", type="Concept")],
                edges=[],
            )
        ],
    )

    assert {node.id for node in graph.nodes} == {"alpha", "beta", "gamma"}
    assert any(
        batch == ["label: Alpha\ntype: Concept", "label: Beta\ntype: Concept"]
        for batch in embedding.batch_calls
    )


def test_combine_into_resolves_only_new_nodes_against_existing(tmp_path):
    store = JsonEmbeddingStore(tmp_path / "embeddings.json")
    existing = KnowledgeGraph(
        nodes=[
            Node(id="openai", label="OpenAI", type="Organization", chunk_ids=["c1"]),
            Node(id="chatgpt", label="ChatGPT", type="Product", chunk_ids=["c1"]),
        ],
        edges=[
            Edge(id="edge-0001", source="openai", target="chatgpt", label="develops", chunk_ids=["c1"]),
        ],
    )
    seeding = SemanticEntityResolver(
        FakeLLM(),
        FakeEmbeddingClient(
            {
                "OpenAI": [1.0, 0.0, 0.0, 0.0],
                "ChatGPT": [0.0, 1.0, 0.0, 0.0],
            }
        ),
        store,
    )
    seeding.combine([existing])

    llm = FakeLLM()
    embedding = FakeEmbeddingClient(
        {
            "OpenAI": [1.0, 0.0, 0.0, 0.0],
            "ChatGPT": [0.0, 1.0, 0.0, 0.0],
            "Anthropic": [0.0, 0.0, 1.0, 0.0],
            "Claude": [0.0, 0.0, 0.0, 1.0],
        }
    )
    resolver = SemanticEntityResolver(llm, embedding, store)

    new_graphs = [
        KnowledgeGraph(
            nodes=[
                Node(id="anthropic", label="Anthropic", type="Organization", chunk_ids=["c2"]),
                Node(id="claude", label="Claude", type="Product", chunk_ids=["c2"]),
            ],
            edges=[
                Edge(id="edge-x", source="anthropic", target="claude", label="develops", chunk_ids=["c2"]),
            ],
        ),
    ]

    graph = resolver.combine_into(existing, new_graphs)

    assert {node.id for node in graph.nodes} == {"openai", "chatgpt", "anthropic", "claude"}
    assert llm.calls == []

    assert not any("label: OpenAI" in call for call in embedding.calls)
    assert not any("label: ChatGPT" in call for call in embedding.calls)


def test_combine_into_preserves_existing_edges_and_appends_new_edges():
    existing = KnowledgeGraph(
        nodes=[
            Node(id="alpha", label="Alpha", type="Concept"),
            Node(id="beta", label="Beta", type="Concept"),
        ],
        edges=[
            Edge(id="edge-0001", source="alpha", target="beta", label="relates_to", chunk_ids=["c1"]),
        ],
    )
    resolver = SemanticEntityResolver(
        FakeLLM(),
        FakeEmbeddingClient(
            {
                "Alpha": [1.0, 0.0, 0.0, 0.0],
                "Beta": [0.0, 1.0, 0.0, 0.0],
                "Gamma": [0.0, 0.0, 1.0, 0.0],
                "Delta": [0.0, 0.0, 0.0, 1.0],
            }
        ),
    )

    new_graphs = [
        KnowledgeGraph(
            nodes=[
                Node(id="gamma", label="Gamma", type="Concept"),
                Node(id="delta", label="Delta", type="Concept"),
            ],
            edges=[
                Edge(id="edge-y", source="delta", target="gamma", label="links_to", chunk_ids=["c2"]),
            ],
        ),
    ]

    graph = resolver.combine_into(existing, new_graphs)

    edge_ids = {edge.id for edge in graph.edges}
    assert "edge-0001" in edge_ids
    new_edge = next(e for e in graph.edges if e.label == "links_to")
    assert new_edge.id == "edge-0002"


def test_combine_into_merges_duplicate_edge_between_existing_and_new():
    existing = KnowledgeGraph(
        nodes=[
            Node(id="alpha", label="Alpha", type="Concept"),
            Node(id="beta", label="Beta", type="Concept"),
        ],
        edges=[
            Edge(id="edge-0001", source="alpha", target="beta", label="relates_to", chunk_ids=["c1"]),
        ],
    )
    resolver = SemanticEntityResolver(
        FakeLLM(),
        FakeEmbeddingClient(
            {
                "Alpha": [1.0, 0.0],
                "Beta": [0.0, 1.0],
            }
        ),
    )

    new_graphs = [
        KnowledgeGraph(
            nodes=[
                Node(id="alpha", label="Alpha", type="Concept", chunk_ids=["c2"]),
                Node(id="beta", label="Beta", type="Concept", chunk_ids=["c2"]),
            ],
            edges=[
                Edge(id="edge-z", source="alpha", target="beta", label="relates_to", chunk_ids=["c2"]),
            ],
        ),
    ]

    graph = resolver.combine_into(existing, new_graphs)

    relates_edges = [e for e in graph.edges if e.label == "relates_to"]
    assert len(relates_edges) == 1
    assert relates_edges[0].chunk_ids == ["c1", "c2"]


def test_combine_into_allocates_edge_ids_after_existing_duplicate_merge():
    existing = KnowledgeGraph(
        nodes=[
            Node(id="alpha", label="Alpha", type="Concept"),
            Node(id="beta", label="Beta", type="Concept"),
            Node(id="gamma", label="Gamma", type="Concept"),
            Node(id="delta", label="Delta", type="Concept"),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="alpha",
                target="beta",
                label="relates_to",
                chunk_ids=["c1"],
            ),
            Edge(
                id="edge-0002",
                source="gamma",
                target="delta",
                label="relates_to",
                chunk_ids=["c1"],
            ),
        ],
    )
    resolver = SemanticEntityResolver(
        FakeLLM(),
        FakeEmbeddingClient(),
        min_candidate_similarity=1.1,
    )

    graph = resolver.combine_into(
        existing,
        [
            KnowledgeGraph(
                nodes=[
                    Node(id="alpha", label="Alpha", type="Concept", chunk_ids=["c2"]),
                    Node(id="beta", label="Beta", type="Concept", chunk_ids=["c2"]),
                    Node(id="epsilon", label="Epsilon", type="Concept"),
                    Node(id="zeta", label="Zeta", type="Concept"),
                ],
                edges=[
                    Edge(
                        id="duplicate",
                        source="alpha",
                        target="beta",
                        label="relates_to",
                        chunk_ids=["c2"],
                    ),
                    Edge(
                        id="new",
                        source="epsilon",
                        target="zeta",
                        label="relates_to",
                    ),
                ],
            )
        ],
    )

    assert [edge.id for edge in graph.edges] == [
        "edge-0001",
        "edge-0002",
        "edge-0003",
    ]
    duplicate = next(
        edge
        for edge in graph.edges
        if edge.source == "alpha" and edge.target == "beta"
    )
    assert duplicate.chunk_ids == ["c1", "c2"]

    graph = resolver.combine_into(
        graph,
        [
            KnowledgeGraph(
                nodes=[
                    Node(id="epsilon", label="Epsilon", type="Concept"),
                    Node(id="delta", label="Delta", type="Concept"),
                ],
                edges=[
                    Edge(
                        id="another-new",
                        source="epsilon",
                        target="delta",
                        label="relates_to",
                    )
                ],
            )
        ],
    )

    edge_ids = [edge.id for edge in graph.edges]
    assert edge_ids == ["edge-0001", "edge-0002", "edge-0003", "edge-0004"]
    assert len(edge_ids) == len(set(edge_ids))
