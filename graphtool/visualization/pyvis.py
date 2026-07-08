import json
from html import escape
from pathlib import Path

from pyvis.network import Network

from graphtool.graph.types import Edge, KnowledgeGraph, Node


def export_graph_html(graph: KnowledgeGraph, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    network = Network(
        height="750px",
        width="100%",
        directed=True,
        cdn_resources="in_line",
    )

    for node in graph.nodes:
        network.add_node(
            node.id,
            label=node.label,
            group=node.type,
            title=_node_title(node),
        )

    for edge in graph.edges:
        network.add_edge(
            edge.source,
            edge.target,
            label=edge.label,
            title=_edge_title(edge),
        )

    network.write_html(str(path), notebook=False)
    return path


def _node_title(node: Node) -> str:
    return _format_title(
        [
            ("id", node.id),
            ("type", node.type),
            ("chunk_ids", node.chunk_ids),
            ("properties", node.properties),
        ]
    )


def _edge_title(edge: Edge) -> str:
    return _format_title(
        [
            ("id", edge.id),
            ("label", edge.label),
            ("chunk_ids", edge.chunk_ids),
            ("properties", edge.properties),
        ]
    )


def _format_title(values: list[tuple[str, object]]) -> str:
    lines = []
    for key, value in values:
        if isinstance(value, str):
            rendered_value = value
        else:
            rendered_value = json.dumps(value, sort_keys=True)
        lines.append(f"<b>{escape(key)}</b>: {escape(rendered_value)}")
    return "<br>".join(lines)
