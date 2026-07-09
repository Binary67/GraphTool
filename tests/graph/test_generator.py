import json
from datetime import datetime, timezone
from logging import Logger
from typing import TypeVar, cast

import pytest
from pydantic import ValidationError

from graphtool.chunking.types import Chunk
from graphtool.graph.generator import (
    _ExtractedEdge,
    _ExtractedKnowledgeGraph,
    _ExtractedNode,
    combine_knowledge_graphs,
    generate_knowledge_graph,
)
from graphtool.graph.taxonomy import JsonTaxonomySuggestionStore
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
    assert "Chunk ID:" not in messages[1].content
    assert "Source:" not in messages[1].content
    assert "Context only, do not extract this as graph content" in messages[1].content
    assert "Heading path: Python" in messages[1].content
    assert "Markdown content:" in messages[1].content
    assert "# Python" in messages[1].content


def test_generate_knowledge_graph_prompt_keeps_metadata_out_of_graph_content():
    fake = FakeLLM([_extracted_graph()])

    generate_knowledge_graph([_chunk()], "doc.md", fake)

    messages, _ = fake.calls[0]
    system_prompt = messages[0].content
    user_prompt = messages[1].content

    assert "important domain entities" in system_prompt
    assert "Do not create nodes for prompt metadata" in system_prompt
    assert "source file paths" in system_prompt
    assert "Table contents can contain useful facts" in system_prompt
    assert "Chunk ID:" not in user_prompt
    assert "Source:" not in user_prompt
    assert "Context only, do not extract this as graph content" in user_prompt
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
                nodes=[_ExtractedNode(id="python", label="Python", type="concept")],
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


def test_generate_knowledge_graph_filters_structural_nodes_and_edges():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="skills", label="Skills", type="feature"),
                    _ExtractedNode(
                        id="workflows",
                        label="Reusable workflows",
                        type="concept",
                    ),
                    _ExtractedNode(
                        id="chunk-wrapper",
                        label="Chunk: Python",
                        type="concept",
                    ),
                    _ExtractedNode(
                        id="table",
                        label="Table",
                        type="unclassified",
                        suggested_type="table",
                    ),
                    _ExtractedNode(id="source-path", label="doc.md", type="concept"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="fact-edge",
                        source="skills",
                        target="workflows",
                        label="used_for",
                    ),
                    _ExtractedEdge(
                        id="table-edge",
                        source="skills",
                        target="table",
                        label="appears_in",
                    ),
                    _ExtractedEdge(
                        id="source-edge",
                        source="source-path",
                        target="skills",
                        label="contains",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert {node.id for node in graph.nodes} == {"skills", "workflows"}
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == "skills"
    assert graph.edges[0].target == "workflows"
    assert graph.edges[0].label == "used_for"


def test_generate_knowledge_graph_keeps_meaningful_document_type():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        id="plugin-guide",
                        label="Plugin guide",
                        type="document",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert [node.id for node in graph.nodes] == ["plugin-guide"]
    assert graph.nodes[0].type == "document"


def test_generate_knowledge_graph_merges_duplicate_nodes_within_chunk():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="openai", label="OpenAI", type="organization"),
                    _ExtractedNode(
                        id="openai",
                        label="OpenAI organization",
                        type="organization",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "openai"
    assert graph.nodes[0].label == "OpenAI"
    assert graph.nodes[0].type == "organization"
    assert graph.nodes[0].aliases == ["OpenAI organization"]
    assert graph.nodes[0].chunk_ids == ["doc-chunk-0000"]


def test_generate_knowledge_graph_renumbers_duplicate_raw_edge_ids_within_chunk():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python", type="concept"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="tool"),
                    _ExtractedNode(id="fastapi", label="FastAPI", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="duplicate-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    ),
                    _ExtractedEdge(
                        id="duplicate-edge",
                        source="fastapi",
                        target="pydantic",
                        label="uses",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert [(edge.id, edge.source, edge.target, edge.label) for edge in graph.edges] == [
        ("edge-0001", "pydantic", "python", "built_for"),
        ("edge-0002", "fastapi", "pydantic", "uses"),
    ]


def test_generate_knowledge_graph_deduplicates_semantic_edges_within_chunk():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python", type="concept"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="first-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    ),
                    _ExtractedEdge(
                        id="second-edge",
                        source="pydantic",
                        target="python",
                        label="built_for",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph([_chunk()], "doc.md", fake)

    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == "pydantic"
    assert graph.edges[0].target == "python"
    assert graph.edges[0].label == "built_for"
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000"]


def test_generate_knowledge_graph_drops_and_records_edges_with_missing_nodes(tmp_path):
    dropped_edges_path = tmp_path / "dropped_edges.jsonl"
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python", type="concept"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="tool"),
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
                        _ExtractedNode(id="python", label="Python", type="concept"),
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


def test_generate_knowledge_graph_logs_generation_counters(tmp_path):
    logger = configure_run_logger(tmp_path / "logs")
    try:
        fake = FakeLLM(
            [
                _extracted_graph(
                    nodes=[
                        _ExtractedNode(id="skills", label="Skills", type="feature"),
                        _ExtractedNode(id="workflow", label="Workflow", type="concept"),
                        _ExtractedNode(
                            id="table",
                            label="Table",
                            type="unclassified",
                            suggested_type="table",
                        ),
                    ],
                    edges=[
                        _ExtractedEdge(
                            id="fact-edge",
                            source="skills",
                            target="workflow",
                            label="supports",
                        ),
                        _ExtractedEdge(
                            id="table-edge",
                            source="skills",
                            target="table",
                            label="appears_in",
                        ),
                    ],
                )
            ]
        )

        generate_knowledge_graph([_chunk()], "doc.md", fake)
        _flush_logger(logger)

        log_files = list((tmp_path / "logs").glob("graphtool-*.log"))
        assert len(log_files) == 1
        text = log_files[0].read_text(encoding="utf-8")
        assert (
            "INFO Generated chunk graph source=doc.md chunk=doc-chunk-0000 "
            "raw_nodes=3 kept_nodes=2 dropped_structural_nodes=1 "
            "raw_edges=2 kept_edges=1 dropped_edges=1"
        ) in text
        assert (
            "INFO Generated document graph source=doc.md chunks=1 "
            "raw_nodes=3 kept_nodes=2 dropped_structural_nodes=1 "
            "raw_edges=2 kept_edges=1 dropped_edges=1 final_nodes=2 final_edges=1"
        ) in text
    finally:
        _close_logger(logger)


def test_generate_knowledge_graph_merges_duplicate_nodes_and_relationships():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(id="python", label="Python", type="concept"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="tool"),
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
                    _ExtractedNode(id="python", label="Python 3", type="concept"),
                    _ExtractedNode(id="pydantic", label="Pydantic", type="tool"),
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
    assert graph.nodes[0].type == "concept"
    assert graph.nodes[0].chunk_ids == ["doc-chunk-0000", "doc-chunk-0001"]
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000", "doc-chunk-0001"]


def test_extracted_knowledge_graph_schema_is_strict_for_openai():
    schema = _ExtractedKnowledgeGraph.model_json_schema()

    assert "properties" not in _ExtractedNode.model_json_schema()["properties"]
    assert "properties" not in _ExtractedEdge.model_json_schema()["properties"]
    assert not _has_additional_properties_true(schema)


def test_extracted_node_rejects_unknown_canonical_type():
    with pytest.raises(ValidationError):
        _ExtractedNode(
            id="marketplace",
            label="Marketplace",
            type="distribution_channel",
        )


def test_extracted_node_requires_suggested_type_for_unclassified():
    with pytest.raises(ValidationError, match="suggested_type is required"):
        _ExtractedNode(
            id="marketplace",
            label="Marketplace",
            type="unclassified",
        )


def test_generate_knowledge_graph_records_taxonomy_suggestions(tmp_path):
    suggestion_store = JsonTaxonomySuggestionStore(tmp_path / "suggestions.json")
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        id="marketplace",
                        label="Marketplace",
                        type="unclassified",
                        suggested_type="distribution channel",
                    )
                ],
                edges=[],
            )
        ]
    )

    generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        fake,
        taxonomy_suggestion_store=suggestion_store,
    )

    records = suggestion_store.load()
    assert len(records) == 1
    assert records[0].suggested_type == "distribution channel"
    assert records[0].normalized_suggested_type == "distribution_channel"
    assert records[0].node_id == "marketplace"
    assert records[0].node_label == "Marketplace"
    assert records[0].current_type == "unclassified"
    assert records[0].source == "doc.md"
    assert records[0].chunk_id == "doc-chunk-0000"
    assert records[0].created_at


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
