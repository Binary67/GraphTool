import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.types import Edge, KnowledgeGraph, Node
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
        return self.combine_into(None, graphs)

    def combine_into(
        self,
        existing: KnowledgeGraph | None,
        graphs: Sequence[KnowledgeGraph],
    ) -> KnowledgeGraph:
        all_graphs = [existing, *graphs] if existing is not None else list(graphs)
        self._relationship_contexts = _build_relationship_contexts(all_graphs)

        canonical_nodes: list[Node] = []
        canonical_by_id: dict[str, Node] = {}
        node_id_map: dict[str, str] = {}

        if existing is not None:
            for node in existing.nodes:
                canonical = _canonicalize_new_node(node)
                canonical_nodes.append(canonical)
                canonical_by_id[canonical.id] = canonical
                node_id_map[node.id] = canonical.id
            existing_edges = [
                _remap_edge(edge, node_id_map) for edge in existing.edges
            ]
        else:
            existing_edges = []

        for graph in graphs:
            for node in graph.nodes:
                canonical_id = self._resolve_node(
                    node,
                    canonical_nodes,
                    canonical_by_id,
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
    ) -> str:
        existing = canonical_by_id.get(node.id)
        if existing is not None:
            self._merge_into(existing, node, canonical_nodes, canonical_by_id)
            return existing.id

        normalized_match = _find_normalized_match(node, canonical_nodes)
        if normalized_match is not None:
            self._merge_into(normalized_match, node, canonical_nodes, canonical_by_id)
            return normalized_match.id

        candidates = self._embedding_candidates(node, canonical_nodes)
        if candidates:
            decision = self._judge_same_entity(node, candidates)
            target_id = _accepted_target_id(
                decision,
                {candidate.id for candidate, _ in candidates},
                self._merge_confidence_threshold,
            )
            if target_id is not None:
                target = canonical_by_id[target_id]
                self._merge_into(
                    target,
                    node,
                    canonical_nodes,
                    canonical_by_id,
                    aliases_to_add=decision.aliases_to_add,
                )
                return target.id

        canonical = _canonicalize_new_node(node)
        canonical_nodes.append(canonical)
        canonical_by_id[canonical.id] = canonical
        return canonical.id

    def _merge_into(
        self,
        existing: Node,
        incoming: Node,
        canonical_nodes: list[Node],
        canonical_by_id: dict[str, Node],
        aliases_to_add: Sequence[str] = (),
    ) -> None:
        merged = _merge_nodes(existing, incoming, aliases_to_add)
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

    def _ensure_embedding(self, node: Node) -> NodeEmbeddingRecord:
        return self._ensure_embeddings([node])[0]

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


def _canonicalize_new_node(node: Node) -> Node:
    return node.model_copy(
        update={
            "aliases": _unique_aliases(node.label, node.aliases),
            "chunk_ids": _extend_unique([], node.chunk_ids),
        }
    )


def _merge_nodes(
    existing: Node,
    incoming: Node,
    aliases_to_add: Sequence[str] = (),
) -> Node:
    alias_additions = []
    if _normalize_name(incoming.label) != _normalize_name(existing.label):
        alias_additions.append(incoming.label)
    alias_additions.extend(incoming.aliases)
    alias_additions.extend(aliases_to_add)

    return existing.model_copy(
        update={
            "aliases": _unique_aliases(
                existing.label,
                [*existing.aliases, *alias_additions],
            ),
            "chunk_ids": _extend_unique(existing.chunk_ids, incoming.chunk_ids),
        }
    )


def _find_normalized_match(node: Node, canonical_nodes: Sequence[Node]) -> Node | None:
    incoming_names = _normalized_names(node)
    if not incoming_names:
        return None

    incoming_type = _normalize_name(node.type)
    for candidate in canonical_nodes:
        if incoming_type != _normalize_name(candidate.type):
            continue
        if incoming_names & _normalized_names(candidate):
            return candidate
    return None


def _same_type_candidates(node: Node, canonical_nodes: Sequence[Node]) -> list[Node]:
    incoming_type = _normalize_name(node.type)
    return [
        candidate
        for candidate in canonical_nodes
        if incoming_type == _normalize_name(candidate.type)
    ]


def _normalized_names(node: Node) -> set[str]:
    return {
        normalized
        for normalized in (_normalize_name(name) for name in [node.label, *node.aliases])
        if normalized
    }


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(normalized.split())


def _unique_aliases(label: str, aliases: Sequence[str]) -> list[str]:
    label_key = _normalize_name(label)
    seen = {label_key}
    unique = []
    for alias in aliases:
        normalized = _normalize_name(alias)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(alias)
    return unique


def _dedupe_remapped_edges(
    graphs: Sequence[KnowledgeGraph],
    node_id_map: Mapping[str, str],
) -> list[Edge]:
    edges_by_key: dict[tuple[str, str, str], Edge] = {}

    for graph in graphs:
        for edge in graph.edges:
            source = node_id_map.get(edge.source, edge.source)
            target = node_id_map.get(edge.target, edge.target)
            remapped = edge.model_copy(update={"source": source, "target": target})
            key = (source, target, remapped.label)
            existing = edges_by_key.get(key)
            if existing is None:
                edges_by_key[key] = remapped
                continue

            edges_by_key[key] = existing.model_copy(
                update={
                    "chunk_ids": _extend_unique(existing.chunk_ids, remapped.chunk_ids)
                }
            )

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
        by_key[key] = match.model_copy(
            update={
                "chunk_ids": _extend_unique(match.chunk_ids, edge.chunk_ids)
            }
        )
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
