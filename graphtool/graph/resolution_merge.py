from collections.abc import Mapping, Sequence

from graphtool.graph.provenance import add_edge_provenance, merge_edges
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node


def same_node_contribution(
    existing: Node,
    node_id: str,
    metadata: GraphMetadata | None,
) -> bool:
    if metadata is None:
        return True
    return any(
        provenance.source == metadata.source
        and provenance.content_hash == metadata.content_hash
        and provenance.node_id == node_id
        for provenance in existing.provenance
    )


def next_node_id(node_id: str, canonical_by_id: Mapping[str, Node]) -> str:
    index = 2
    while True:
        candidate = f"{node_id}::{index:04d}"
        if candidate not in canonical_by_id:
            return candidate
        index += 1


def dedupe_remapped_edges(
    graphs: Sequence[KnowledgeGraph],
    node_id_map: Mapping[str, str],
) -> list[Edge]:
    edges_by_key: dict[tuple[str, str, str], Edge] = {}
    for graph in graphs:
        for edge in graph.edges:
            contributed = add_edge_provenance(edge, graph.metadata)
            source = node_id_map.get(edge.source, edge.source)
            target = node_id_map.get(edge.target, edge.target)
            remapped = contributed.model_copy(
                update={"source": source, "target": target}
            )
            key = (source, target, remapped.label)
            existing = edges_by_key.get(key)
            edges_by_key[key] = (
                remapped
                if existing is None
                else merge_edges(existing, remapped)
            )
    return list(edges_by_key.values())


def remap_edge(edge: Edge, node_id_map: Mapping[str, str]) -> Edge:
    return edge.model_copy(
        update={
            "source": node_id_map.get(edge.source, edge.source),
            "target": node_id_map.get(edge.target, edge.target),
        }
    )


def merge_edge_sets(
    existing: Sequence[Edge],
    incoming: Sequence[Edge],
) -> list[Edge]:
    by_key = {
        (edge.source, edge.target, edge.label): edge
        for edge in existing
    }
    used_ids = {edge.id for edge in existing}
    next_index = 1
    for edge in incoming:
        key = (edge.source, edge.target, edge.label)
        match = by_key.get(key)
        if match is not None:
            by_key[key] = merge_edges(match, edge)
            continue
        edge_id, next_index = _next_edge_id(used_ids, next_index)
        by_key[key] = edge.model_copy(update={"id": edge_id})
    return list(by_key.values())


def _next_edge_id(used_ids: set[str], start_index: int) -> tuple[str, int]:
    index = start_index
    while True:
        edge_id = f"edge-{index:04d}"
        index += 1
        if edge_id not in used_ids:
            used_ids.add(edge_id)
            return edge_id, index
