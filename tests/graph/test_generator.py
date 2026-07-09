import json
from datetime import datetime, timezone
from logging import Logger
from typing import TypeVar, cast

from graphtool.chunking.types import Chunk
from graphtool.graph.generator import (
    _ExtractedEdge,
    _ExtractedKnowledgeGraph,
    _ExtractedNode,
    combine_knowledge_graphs,
    generate_knowledge_graph,
)
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage
from graphtool.run_logging import configure_run_logger

T = TypeVar("T")


class FakeLLM:
    def __init__(self, responses: list[_ExtractedKnowledgeGraph]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return cast(T, self.responses[len(self.calls) - 1])


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


def _extracted_graph(
    nodes: list[_ExtractedNode] | None = None,
    edges: list[_ExtractedEdge] | None = None,
) -> _ExtractedKnowledgeGraph:
    return _ExtractedKnowledgeGraph(nodes=nodes or [], edges=edges or [])


def test_generate_knowledge_graph_invokes_structured_generation_per_chunk():
    fake = FakeLLM([_extracted_graph(), _extracted_graph()])
    chunks = [
        _chunk("doc-chunk-0000", 0),
        _chunk("doc-chunk-0001", 1, "## Pydantic\nValidation.", ["Python", "Pydantic"]),
    ]

    generate_knowledge_graph(chunks, "doc.md", fake)

    assert len(fake.calls) == 2
    messages, response_model = fake.calls[0]
    assert response_model is _ExtractedKnowledgeGraph
    assert messages[0].role == "system"
    assert "knowledge graph" in messages[0].content
    assert messages[1].role == "user"
    assert "Chunk ID: doc-chunk-0000" in messages[1].content
    assert "Source: doc.md" in messages[1].content
    assert "Heading path: Python" in messages[1].content
    assert "# Python" in messages[1].content


def test_generate_knowledge_graph_prompt_keeps_metadata_out_of_graph_content():
    fake = FakeLLM([_extracted_graph()])

    generate_knowledge_graph([_chunk()], "doc.md", fake)

    messages, _ = fake.calls[0]
    system_prompt = messages[0].content
    user_prompt = messages[1].content

    assert "important named entities or concise noun phrases" in system_prompt
    assert "Do not create nodes for full actions" in system_prompt
    assert "headings, chunk ids, source paths, URLs" in system_prompt
    assert (
        "Use prompt metadata such as Chunk ID, Source, and Heading path only as context"
        in system_prompt
    )
    assert "never represent that metadata as nodes or edges" in system_prompt
    assert "Express actions, predicates" in system_prompt
    assert "Chunk ID: doc-chunk-0000" in user_prompt
    assert "Source: doc.md" in user_prompt
    assert "Heading path: Python" in user_prompt


def test_generate_knowledge_graph_attaches_metadata():
    fake = FakeLLM([_extracted_graph()])

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert graph.metadata is not None
    assert graph.metadata.source == "doc.md"
    assert graph.metadata.created_at <= datetime.now(timezone.utc)


def test_generate_knowledge_graph_attaches_chunk_ids_to_nodes_and_edges():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[_ExtractedNode(id="python", label="Python", type="Language")],
                edges=[
                    _ExtractedEdge(
                        id="e1",
                        source="python",
                        target="python",
                        label="self",
                    )
                ],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert graph.nodes[0].chunk_ids == ["doc-chunk-0000"]
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000"]


def test_generate_knowledge_graph_drops_and_records_edges_with_missing_nodes(tmp_path):
    dropped_edges_path = tmp_path / "dropped_edges.jsonl"
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python", type="Language"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="Library"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="valid-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    ),
                    _ExtractedEdge(
                        id="missing-edge",
                        source="pydantic",
                        target="missing-node",
                        label="mentions",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        fake,
        dropped_edges_path=dropped_edges_path,
    )

    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == "pydantic"
    assert graph.edges[0].target == "python"

    records = [
        json.loads(line)
        for line in dropped_edges_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["source"] == "doc.md"
    assert records[0]["chunk_id"] == "doc-chunk-0000"
    assert records[0]["edge_id"] == "missing-edge"
    assert records[0]["label"] == "mentions"
    assert records[0]["edge_source"] == "pydantic"
    assert records[0]["edge_target"] == "missing-node"
    assert records[0]["missing"] == ["target"]
    assert records[0]["created_at"]


def test_generate_knowledge_graph_logs_dropped_edges_to_run_log(tmp_path):
    logger = configure_run_logger(tmp_path / "logs")
    try:
        fake = FakeLLM(
            [
                _extracted_graph(
                    nodes=[
                        _ExtractedNode(id="python", label="Python", type="Language"),
                    ],
                    edges=[
                        _ExtractedEdge(
                            id="missing-edge",
                            source="python",
                            target="missing-node",
                            label="mentions",
                        )
                    ],
                )
            ]
        )

        generate_knowledge_graph([_chunk()], "doc.md", fake)
        _flush_logger(logger)

        log_files = list((tmp_path / "logs").glob("graphtool-*.log"))
        assert len(log_files) == 1
        assert (
            "WARNING Skipped extracted edge missing-edge in doc-chunk-0000: "
            "missing target node missing-node"
        ) in log_files[0].read_text(encoding="utf-8")
    finally:
        _close_logger(logger)


def test_generate_knowledge_graph_merges_duplicate_nodes_and_relationships():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python", type="Language"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="Library"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="first-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    )
                ],
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python 3", type="Version"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="Library"),
                ],
                edges=[
                    _ExtractedEdge(
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


def test_extracted_knowledge_graph_schema_is_strict_for_openai():
    schema = _ExtractedKnowledgeGraph.model_json_schema()

    assert "properties" not in _ExtractedNode.model_json_schema()["properties"]
    assert "properties" not in _ExtractedEdge.model_json_schema()["properties"]
    assert not _has_additional_properties_true(schema)


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


def _has_additional_properties_true(value) -> bool:
    if isinstance(value, dict):
        return any(
            (key == "additionalProperties" and child is True)
            or _has_additional_properties_true(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_additional_properties_true(child) for child in value)
    return False


def _flush_logger(logger: Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _close_logger(logger: Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
