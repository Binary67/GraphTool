import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.provenance import (
    add_edge_provenance,
    add_node_provenance,
    canonicalize_node,
    merge_edges,
    merge_nodes,
)
from graphtool.graph.taxonomy import UNCLASSIFIED_NODE_TYPE, normalize_type_name
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient, LLMClient
from graphtool.llm.types import LLMMessage

DEFAULT_TOP_K = 10
DEFAULT_MERGE_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MIN_CANDIDATE_SIMILARITY = 0.80

ENTITY_RESOLUTION_SYSTEM_PROMPT = (
    "You decide whether a new knowledge graph node refers to the same real-world "
    "entity as one of the provided candidate nodes. Merge only when they are the "
    "same entity, not when they are merely related. Keep organizations, products, "
    "services, APIs, and deployments distinct unless the names clearly refer to "
    "the same entity."
)


class EmbeddingStore(Protocol):
    def load(self) -> dict[str, NodeEmbeddingRecord]:
        ...

    def save(self, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        ...


class EntityResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["merge", "new"]
    target_node_id: str | None = None
    confidence: float = 0.0
    canonical_label: str | None = None
    aliases_to_add: list[str] = Field(default_factory=list)


class SemanticEntityResolver:
    def __init__(
        self,
        llm: LLMClient,
        embedding_client: EmbeddingClient,
        embedding_store: EmbeddingStore | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        merge_confidence_threshold: float = DEFAULT_MERGE_CONFIDENCE_THRESHOLD,
        min_candidate_similarity: float = DEFAULT_MIN_CANDIDATE_SIMILARITY,
    ) -> None:
        self._llm = llm
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store
        self._records = embedding_store.load() if embedding_store is not None else {}
        self._top_k = top_k
        self._merge_confidence_threshold = merge_confidence_threshold
        self._min_candidate_similarity = min_candidate_similarity
        self._relationship_contexts: dict[str, list[str]] = {}

    def combine(self, graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
        return self._combine(None, graphs, resolve_within_graph=True)

    def combine_into(
        self,
        existing: KnowledgeGraph | None,
        graphs: Sequence[KnowledgeGraph],
    ) -> KnowledgeGraph:
        return self._combine(existing, graphs, resolve_within_graph=False)

    def _combine(
        self,
        existing: KnowledgeGraph | None,
        graphs: Sequence[KnowledgeGraph],
        *,
        resolve_within_graph: bool,
    ) -> KnowledgeGraph:
        all_graphs = [existing, *graphs] if existing is not None else list(graphs)
        self._relationship_contexts = _build_relationship_contexts(all_graphs)

        canonical_nodes: list[Node] = []
        canonical_by_id: dict[str, Node] = {}
        node_id_map: dict[str, str] = {}

        if existing is not None:
            for node in existing.nodes:
                canonical = canonicalize_node(node)
                canonical_nodes.append(canonical)
                canonical_by_id[canonical.id] = canonical
                node_id_map[node.id] = canonical.id
            existing_edges = [
                _remap_edge(edge, node_id_map) for edge in existing.edges
            ]
        else:
            existing_edges = []

        for graph in graphs:
            candidate_ids = (
                None if resolve_within_graph else set(canonical_by_id)
            )
            for node in graph.nodes:
                candidates = (
                    canonical_nodes
                    if candidate_ids is None
                    else [
                        candidate
                        for candidate in canonical_nodes
                        if candidate.id in candidate_ids
                    ]
                )
                canonical_id = self._resolve_node(
                    node,
                    canonical_nodes,
                    canonical_by_id,
                    candidates,
                    graph.metadata,
                )
                node_id_map[node.id] = canonical_id

        edges = _dedupe_remapped_edges(graphs, node_id_map)
        edges = _merge_edges(existing_edges, edges)
        graph = KnowledgeGraph(nodes=canonical_nodes, edges=edges)

        if self._embedding_store is not None:
            self._ensure_embeddings(canonical_nodes)
            self._records = {
                node.id: self._records[node.id]
                for node in canonical_nodes
            }
            self._embedding_store.save(self._records)

        return graph

    def _resolve_node(
        self,
        node: Node,
        canonical_nodes: list[Node],
        canonical_by_id: dict[str, Node],
        candidates: Sequence[Node],
        metadata: GraphMetadata | None,
    ) -> str:
        existing = canonical_by_id.get(node.id)
        if existing is not None and _same_node_contribution(
            existing,
            node.id,
            metadata,
        ):
            self._merge_into(
                existing,
                node,
                canonical_nodes,
                canonical_by_id,
                metadata=metadata,
            )
            return existing.id

        normalized_match = _find_normalized_match(node, candidates)
        if normalized_match is not None:
            self._merge_into(
                normalized_match,
                node,
                canonical_nodes,
                canonical_by_id,
                metadata=metadata,
            )
            return normalized_match.id

        embedding_candidates = self._embedding_candidates(node, candidates)
        if embedding_candidates:
            decision = self._judge_same_entity(node, embedding_candidates)
            target_id = _accepted_target_id(
                decision,
                {candidate.id for candidate, _ in embedding_candidates},
                self._merge_confidence_threshold,
            )
            if target_id is not None:
                target = canonical_by_id[target_id]
                self._merge_into(
                    target,
                    node,
                    canonical_nodes,
                    canonical_by_id,
                    metadata=metadata,
                    aliases_to_add=decision.aliases_to_add,
                )
                return target.id

        canonical = canonicalize_node(add_node_provenance(node, metadata))
        if canonical.id in canonical_by_id:
            canonical = canonical.model_copy(
                update={"id": _next_node_id(canonical.id, canonical_by_id)}
            )
        canonical_nodes.append(canonical)
        canonical_by_id[canonical.id] = canonical
        return canonical.id

    def _merge_into(
        self,
        existing: Node,
        incoming: Node,
        canonical_nodes: list[Node],
        canonical_by_id: dict[str, Node],
        metadata: GraphMetadata | None,
        aliases_to_add: Sequence[str] = (),
    ) -> None:
        if metadata is None:
            merged = merge_nodes(existing, incoming, aliases_to_add)
        else:
            contributed = add_node_provenance(
                incoming,
                metadata,
                aliases_to_add,
            )
            merged = merge_nodes(existing, contributed)
        index = canonical_nodes.index(existing)
        canonical_nodes[index] = merged
        canonical_by_id[merged.id] = merged
        self._records.pop(merged.id, None)

    def _embedding_candidates(
        self,
        node: Node,
        canonical_nodes: Sequence[Node],
    ) -> list[tuple[Node, float]]:
        candidate_nodes = _same_type_candidates(node, canonical_nodes)
        if not candidate_nodes:
            return []

        incoming_text = node_embedding_text(
            node,
            self._relationship_contexts.get(node.id, []),
        )
        incoming_vector = self._embedding_client.embed_texts([incoming_text])[0]
        candidate_records = self._ensure_embeddings(candidate_nodes)
        scored = []
        for candidate, record in zip(candidate_nodes, candidate_records, strict=True):
            score = _cosine_similarity(incoming_vector, record.vector)
            if score >= self._min_candidate_similarity:
                scored.append((candidate, score))

        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[: self._top_k]

    def _ensure_embeddings(self, nodes: Sequence[Node]) -> list[NodeEmbeddingRecord]:
        records: list[NodeEmbeddingRecord | None] = []
        missing: list[tuple[int, Node, str, str]] = []
        embedding_model = self._embedding_client.embedding_model

        for node in nodes:
            text = node_embedding_text(
                node,
                self._relationship_contexts.get(node.id, []),
            )
            text_hash = embedding_input_hash(text)
            existing = self._records.get(node.id)
            if (
                existing is not None
                and existing.embedding_model == embedding_model
                and existing.embedding_input_hash == text_hash
            ):
                records.append(existing)
                continue

            records.append(None)
            missing.append((len(records) - 1, node, text_hash, text))

        if missing:
            vectors = self._embedding_client.embed_texts(
                [text for _, _, _, text in missing]
            )
            for (index, node, text_hash, _), vector in zip(
                missing,
                vectors,
                strict=True,
            ):
                record = NodeEmbeddingRecord(
                    node_id=node.id,
                    embedding_model=embedding_model,
                    embedding_input_hash=text_hash,
                    vector=vector,
                )
                self._records[node.id] = record
                records[index] = record

        return [record for record in records if record is not None]

    def _judge_same_entity(
        self,
        node: Node,
        candidates: Sequence[tuple[Node, float]],
    ) -> EntityResolutionDecision:
        payload = {
            "incoming": _node_payload(node),
            "candidates": [
                {
                    **_node_payload(candidate),
                    "similarity": round(score, 6),
                }
                for candidate, score in candidates
            ],
            "rules": [
                "Return merge only if the incoming node and target candidate are the same entity.",
                "Do not merge an organization with its product, API, service, deployment, or partner.",
                "When uncertain, return new.",
            ],
        }
        messages = [
            LLMMessage(role="system", content=ENTITY_RESOLUTION_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    "Resolve the incoming node against the candidates. "
                    "Return the structured decision only.\n\n"
                    f"{json.dumps(payload, indent=2, sort_keys=True)}"
                ),
            ),
        ]
        return self._llm.generate_structured(messages, EntityResolutionDecision)


def node_embedding_text(node: Node, relationship_context: Sequence[str] = ()) -> str:
    parts = [
        f"label: {node.label}",
        f"type: {node.type}",
    ]
    if node.suggested_type:
        parts.append(f"suggested_type: {node.suggested_type}")
    if node.aliases:
        parts.append(f"aliases: {', '.join(node.aliases)}")
    if node.properties:
        parts.append(f"properties: {json.dumps(node.properties, sort_keys=True)}")
    if relationship_context:
        relationships = "; ".join(_extend_unique([], list(relationship_context)))
        parts.append(f"relationships: {relationships}")
    return "\n".join(parts)


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _accepted_target_id(
    decision: EntityResolutionDecision,
    candidate_ids: set[str],
    merge_confidence_threshold: float,
) -> str | None:
    if decision.decision != "merge":
        return None
    if decision.target_node_id not in candidate_ids:
        return None
    if decision.confidence < merge_confidence_threshold:
        return None
    return decision.target_node_id


def _same_node_contribution(
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


def _next_node_id(node_id: str, canonical_by_id: Mapping[str, Node]) -> str:
    index = 2
    while True:
        candidate = f"{node_id}::{index:04d}"
        if candidate not in canonical_by_id:
            return candidate
        index += 1


def _find_normalized_match(node: Node, canonical_nodes: Sequence[Node]) -> Node | None:
    incoming_names = _normalized_names(node)
    if not incoming_names:
        return None

    for candidate in canonical_nodes:
        if not _comparable_node_types(node, candidate):
            continue
        if incoming_names & _normalized_names(candidate):
            return candidate
    return None


def _same_type_candidates(node: Node, canonical_nodes: Sequence[Node]) -> list[Node]:
    return [
        candidate
        for candidate in canonical_nodes
        if _comparable_node_types(node, candidate)
    ]


def _normalized_names(node: Node) -> set[str]:
    return {
        normalized
        for normalized in (_normalize_name(name) for name in [node.label, *node.aliases])
        if normalized
    }


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(_singularize_word(word) for word in normalized.split())


def _singularize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if (
        len(word) > 4
        and word.endswith(("ches", "shes", "sses", "xes", "zes"))
    ):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _comparable_node_types(left: Node, right: Node) -> bool:
    left_type = normalize_type_name(left.type)
    right_type = normalize_type_name(right.type)
    return (
        left_type == right_type
        or left_type == UNCLASSIFIED_NODE_TYPE
        or right_type == UNCLASSIFIED_NODE_TYPE
    )


def _dedupe_remapped_edges(
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
            if existing is None:
                edges_by_key[key] = remapped
                continue

            edges_by_key[key] = merge_edges(existing, remapped)

    return list(edges_by_key.values())


def _remap_edge(edge: Edge, node_id_map: Mapping[str, str]) -> Edge:
    source = node_id_map.get(edge.source, edge.source)
    target = node_id_map.get(edge.target, edge.target)
    return edge.model_copy(update={"source": source, "target": target})


def _merge_edges(existing: Sequence[Edge], incoming: Sequence[Edge]) -> list[Edge]:
    by_key: dict[tuple[str, str, str], Edge] = {}
    used_ids = {edge.id for edge in existing}
    next_index = 1
    for edge in existing:
        by_key[(edge.source, edge.target, edge.label)] = edge
    for edge in incoming:
        key = (edge.source, edge.target, edge.label)
        match = by_key.get(key)
        if match is None:
            edge_id, next_index = _next_edge_id(used_ids, next_index)
            by_key[key] = edge.model_copy(update={"id": edge_id})
            continue
        by_key[key] = merge_edges(match, edge)
    return list(by_key.values())


def _next_edge_id(used_ids: set[str], start_index: int) -> tuple[str, int]:
    index = start_index
    while True:
        edge_id = f"edge-{index:04d}"
        index += 1
        if edge_id not in used_ids:
            used_ids.add(edge_id)
            return edge_id, index


def _build_relationship_contexts(
    graphs: Sequence[KnowledgeGraph],
) -> dict[str, list[str]]:
    contexts: dict[str, list[str]] = {}
    for graph in graphs:
        nodes_by_id = {node.id: node for node in graph.nodes}
        for edge in graph.edges:
            source = nodes_by_id.get(edge.source)
            target = nodes_by_id.get(edge.target)
            if source is not None and target is not None:
                contexts.setdefault(source.id, []).append(
                    f"outgoing {edge.label} to {target.label} ({target.type})"
                )
                contexts.setdefault(target.id, []).append(
                    f"incoming {edge.label} from {source.label} ({source.type})"
                )
    return contexts


def _node_payload(node: Node) -> dict[str, object]:
    return {
        "id": node.id,
        "label": node.label,
        "type": node.type,
        "suggested_type": node.suggested_type,
        "aliases": node.aliases,
        "properties": node.properties,
    }


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _extend_unique(values: list[str], additions: Sequence[str]) -> list[str]:
    merged = list(values)
    for addition in additions:
        if addition not in merged:
            merged.append(addition)
    return merged
