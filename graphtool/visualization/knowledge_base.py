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

    for graph in graphs:
        if graph.metadata is None:
            raise ValueError("Cannot visualize graph without metadata.source.")

        paths.append(
            export_graph_html(
                graph,
                path / "documents" / f"{source_key(graph.metadata.source)}.html",
            )
        )

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
