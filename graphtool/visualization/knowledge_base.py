from pathlib import Path

from graphtool.graph.generator import combine_knowledge_graphs
from graphtool.graph.json_store import JsonGraphStore
from graphtool.source import source_key
from graphtool.visualization.pyvis import export_graph_html


def export_knowledge_base_visualizations(
    graph_store: JsonGraphStore,
    output_dir: str | Path,
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
            combine_knowledge_graphs(graphs),
            path / "knowledge_graph.html",
        )
    )
    return paths
