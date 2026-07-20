from collections.abc import Sequence

from graphtool.graph.provenance import (
    add_edge_provenance,
    add_node_provenance,
    merge_edges,
    merge_nodes,
)
from graphtool.graph.types import Edge, KnowledgeGraph, Node


def combine_knowledge_graphs(graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
    nodes_by_id: dict[str, Node] = {}
    edges_by_key: dict[tuple[str, str, str], Edge] = {}
    used_edge_ids: set[str] = set()
    next_edge_index = 1

    for graph in graphs:
        for node in graph.nodes:
            node = add_node_provenance(node, graph.metadata)
            existing_node = nodes_by_id.get(node.id)
            nodes_by_id[node.id] = (
                node
                if existing_node is None
                else merge_nodes(existing_node, node)
            )

        for edge in graph.edges:
            edge = add_edge_provenance(edge, graph.metadata)
            key = (edge.source, edge.target, edge.label)
            existing_edge = edges_by_key.get(key)
            if existing_edge is not None:
                edges_by_key[key] = merge_edges(existing_edge, edge)
                continue

            if not edge.provenance or edge.id in used_edge_ids:
                edge_id, next_edge_index = _next_edge_id(
                    used_edge_ids,
                    next_edge_index,
                )
                edge = edge.model_copy(update={"id": edge_id})
            else:
                used_edge_ids.add(edge.id)
            edges_by_key[key] = edge

    return KnowledgeGraph(
        nodes=list(nodes_by_id.values()),
        edges=list(edges_by_key.values()),
    )


def _next_edge_id(used_ids: set[str], start_index: int) -> tuple[str, int]:
    index = start_index
    while True:
        edge_id = f"edge-{index:04d}"
        index += 1
        if edge_id not in used_ids:
            used_ids.add(edge_id)
            return edge_id, index
