from datetime import datetime, timezone
from typing import TypeVar

from graphtool.graph.generator import generate_knowledge_graph
from graphtool.graph.types import GraphMetadata, KnowledgeGraph, Node, Edge
from graphtool.llm.types import LLMMessage

T = TypeVar("T")


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return KnowledgeGraph(
            nodes=[Node(id="python", label="Python", type="Language")],
            edges=[Edge(id="e1", source="python", target="python", label="self")],
        )


def test_generate_knowledge_graph_invokes_structured_generation():
    fake = FakeLLM()

    graph = generate_knowledge_graph("# Python\nA dynamic language.", "python.md", fake)

    assert len(fake.calls) == 1
    messages, response_model = fake.calls[0]
    assert response_model is KnowledgeGraph
    assert messages[0].role == "system"
    assert "knowledge graph" in messages[0].content
    assert messages[1].role == "user"
    assert "# Python" in messages[1].content


def test_generate_knowledge_graph_attaches_metadata():
    fake = FakeLLM()

    graph = generate_knowledge_graph("# Python\nA dynamic language.", "python.md", fake)

    assert graph.metadata is not None
    assert graph.metadata.source == "python.md"
    assert graph.metadata.created_at <= datetime.now(timezone.utc)


def test_generate_knowledge_graph_preserves_llm_nodes_and_edges():
    fake = FakeLLM()

    graph = generate_knowledge_graph("irrelevant", "doc.md", fake)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "python"
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "python"