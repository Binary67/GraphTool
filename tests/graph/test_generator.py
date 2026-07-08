from datetime import datetime, timezone
from typing import TypeVar

from graphtool.chunking.types import Chunk
from graphtool.graph.generator import combine_knowledge_graphs, generate_knowledge_graph
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage

T = TypeVar("T")


class FakeLLM:
    def __init__(self, responses: list[KnowledgeGraph]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return self.responses[len(self.calls) - 1]


def _chunk(
    chunk_id: str = "doc-chunk-0000",
    index: int = 0,
    text: str = "# Python\nA dynamic language.",
    heading_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        source="doc.md",
        index=index,
        text=text,
        heading_path=heading_path or ["Python"],
    )


def test_generate_knowledge_graph_invokes_structured_generation_per_chunk():
    fake = FakeLLM([KnowledgeGraph(nodes=[], edges=[]), KnowledgeGraph(nodes=[], edges=[])])
    chunks = [
        _chunk("doc-chunk-0000", 0),
        _chunk("doc-chunk-0001", 1, "## Pydantic\nValidation.", ["Python", "Pydantic"]),
    ]

    generate_knowledge_graph(chunks, "doc.md", fake)

    assert len(fake.calls) == 2
    messages, response_model = fake.calls[0]
    assert response_model is KnowledgeGraph
    assert messages[0].role == "system"
    assert "knowledge graph" in messages[0].content
    assert messages[1].role == "user"
    assert "Chunk ID: doc-chunk-0000" in messages[1].content
    assert "Source: doc.md" in messages[1].content
    assert "Heading path: Python" in messages[1].content
    assert "# Python" in messages[1].content


def test_generate_knowledge_graph_attaches_metadata():
    fake = FakeLLM([KnowledgeGraph(nodes=[], edges=[])])

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert graph.metadata is not None
    assert graph.metadata.source == "doc.md"
    assert graph.metadata.created_at <= datetime.now(timezone.utc)


def test_generate_knowledge_graph_attaches_chunk_ids_to_nodes_and_edges():
    fake = FakeLLM(
        [
            KnowledgeGraph(
                nodes=[Node(id="python", label="Python", type="Language")],
                edges=[Edge(id="e1", source="python", target="python", label="self")],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert graph.nodes[0].chunk_ids == ["doc-chunk-0000"]
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000"]


def test_generate_knowledge_graph_merges_duplicate_nodes_and_relationships():
    fake = FakeLLM(
        [
            KnowledgeGraph(
                nodes=[
                    Node(id="python", label="Python", type="Language"),
                    Node(id="pydantic", label="Pydantic", type="Library"),
                ],
                edges=[
                    Edge(
                        id="first-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    )
                ],
            ),
            KnowledgeGraph(
                nodes=[
                    Node(id="python", label="Python 3", type="Version"),
                    Node(id="pydantic", label="Pydantic", type="Library"),
                ],
                edges=[
                    Edge(
                        id="second-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    )
                ],
            ),
        ]
    )
    chunks = [
        _chunk("doc-chunk-0000", 0),
        _chunk("doc-chunk-0001", 1, "## Pydantic\nValidation.", ["Python", "Pydantic"]),
    ]

    graph = generate_knowledge_graph(chunks, "doc.md", fake)

    assert len(graph.nodes) == 2
    assert graph.nodes[0].id == "python"
    assert graph.nodes[0].label == "Python"
    assert graph.nodes[0].type == "Language"
    assert graph.nodes[0].chunk_ids == ["doc-chunk-0000", "doc-chunk-0001"]
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000", "doc-chunk-0001"]


def test_combine_knowledge_graphs_merges_multiple_document_graphs():
    graph = combine_knowledge_graphs(
        [
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="python",
                        label="Python",
                        type="Language",
                        chunk_ids=["first-chunk-0000"],
                    )
                ],
                edges=[
                    Edge(
                        id="first-edge",
                        source="python",
                        target="python",
                        label="mentions",
                        chunk_ids=["first-chunk-0000"],
                    )
                ],
            ),
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="python",
                        label="Python 3",
                        type="Version",
                        chunk_ids=["second-chunk-0000"],
                    )
                ],
                edges=[
                    Edge(
                        id="second-edge",
                        source="python",
                        target="python",
                        label="mentions",
                        chunk_ids=["second-chunk-0000"],
                    )
                ],
            ),
        ]
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].chunk_ids == ["first-chunk-0000", "second-chunk-0000"]
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["first-chunk-0000", "second-chunk-0000"]
