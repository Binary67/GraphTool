from pathlib import Path

from graphtool.graph.generator import combine_knowledge_graphs
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.types import KnowledgeGraph
from graphtool.source import source_key
from graphtool.visualization.pyvis import export_graph_html


def export_knowledge_base_visualizations(
    graph_store: JsonGraphStore,
    output_dir: str | Path,
    *,
    knowledge_base_store: JsonKnowledgeBaseStore | None = None,
) -> list[Path]:
    path = Path(output_dir)
    graphs = graph_store.load_all()
    paths = []
    expected_document_paths = set()

    for graph in graphs:
        if graph.metadata is None:
            raise ValueError("Cannot visualize graph without metadata.source.")

        document_path = (
            path / "documents" / f"{source_key(graph.metadata.source)}.html"
        )
        expected_document_paths.add(document_path.resolve())
        paths.append(export_graph_html(graph, document_path))

    documents_path = path / "documents"
    if documents_path.exists():
        for existing_path in documents_path.glob("*.html"):
            if existing_path.resolve() not in expected_document_paths:
                existing_path.unlink()

    paths.append(
        export_graph_html(
            _load_combined_graph(graphs, knowledge_base_store),
            path / "knowledge_graph.html",
        )
    )
    return paths


def _load_combined_graph(
    graphs: list[KnowledgeGraph],
    knowledge_base_store: JsonKnowledgeBaseStore | None,
) -> KnowledgeGraph:
    if knowledge_base_store is None:
        return combine_knowledge_graphs(graphs)
    if knowledge_base_store.exists():
        return knowledge_base_store.load()

    graph = combine_knowledge_graphs(graphs)
    knowledge_base_store.save(graph)
    return graph
