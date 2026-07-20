import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.entity_matching import same_type_candidates
from graphtool.graph.types import KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient


class EmbeddingStore(Protocol):
    def load(self) -> dict[str, NodeEmbeddingRecord]:
        ...

    def save(self, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        ...


class ResolutionEmbeddings:
    def __init__(
        self,
        client: EmbeddingClient,
        store: EmbeddingStore | None,
    ) -> None:
        self._client = client
        self._store = store
        self._records = store.load() if store is not None else {}
        self._relationship_contexts: dict[str, list[str]] = {}
        self._prefetched_vectors: dict[tuple[str, str], list[float]] = {}

    def prepare(
        self,
        graphs: Sequence[KnowledgeGraph],
        incoming_nodes: Sequence[Node],
    ) -> None:
        self._relationship_contexts = _build_relationship_contexts(graphs)
        self._prefetched_vectors = {}
        self._prefetch(incoming_nodes)

    def candidates(
        self,
        node: Node,
        canonical_nodes: Sequence[Node],
        *,
        min_similarity: float,
        top_k: int,
    ) -> list[tuple[Node, float]]:
        candidate_nodes = same_type_candidates(node, canonical_nodes)
        if not candidate_nodes:
            return []

        incoming_vector = self._ensure([node])[0].vector
        candidate_records = self._ensure(candidate_nodes)
        scored = []
        for candidate, record in zip(
            candidate_nodes,
            candidate_records,
            strict=True,
        ):
            score = _cosine_similarity(incoming_vector, record.vector)
            if score >= min_similarity:
                scored.append((candidate, score))

        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:top_k]

    def invalidate(self, node_id: str) -> None:
        self._records.pop(node_id, None)

    def finalize(self, canonical_nodes: Sequence[Node]) -> None:
        if self._store is None:
            return
        self._ensure(canonical_nodes)
        self._records = {
            node.id: self._records[node.id]
            for node in canonical_nodes
        }
        self._store.save(self._records)

    def _prefetch(self, nodes: Sequence[Node]) -> None:
        embedding_model = self._client.embedding_model
        missing_by_key: dict[tuple[str, str], str] = {}

        for node in nodes:
            text = node_embedding_text(
                node,
                self._relationship_contexts.get(node.id, []),
            )
            text_hash = embedding_input_hash(text)
            key = (embedding_model, text_hash)
            if key in self._prefetched_vectors:
                continue

            existing = self._records.get(node.id)
            if (
                existing is not None
                and existing.embedding_model == embedding_model
                and existing.embedding_input_hash == text_hash
            ):
                self._prefetched_vectors[key] = existing.vector
                continue
            missing_by_key.setdefault(key, text)

        if missing_by_key:
            keys = list(missing_by_key)
            vectors = self._client.embed_texts(
                [missing_by_key[key] for key in keys]
            )
            self._prefetched_vectors.update(zip(keys, vectors, strict=True))

    def _ensure(self, nodes: Sequence[Node]) -> list[NodeEmbeddingRecord]:
        records: list[NodeEmbeddingRecord | None] = []
        missing: list[tuple[int, Node, str, str]] = []
        embedding_model = self._client.embedding_model

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

            prefetched = self._prefetched_vectors.get((embedding_model, text_hash))
            if prefetched is not None:
                record = NodeEmbeddingRecord(
                    node_id=node.id,
                    embedding_model=embedding_model,
                    embedding_input_hash=text_hash,
                    vector=prefetched,
                )
                self._records[node.id] = record
                records.append(record)
                continue

            records.append(None)
            missing.append((len(records) - 1, node, text_hash, text))

        if missing:
            vectors = self._client.embed_texts(
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


def node_embedding_text(
    node: Node,
    relationship_context: Sequence[str] = (),
) -> str:
    parts = [f"label: {node.label}", f"type: {node.type}"]
    if node.suggested_type:
        parts.append(f"suggested_type: {node.suggested_type}")
    if node.aliases:
        parts.append(f"aliases: {', '.join(node.aliases)}")
    if node.properties:
        parts.append(f"properties: {json.dumps(node.properties, sort_keys=True)}")
    if relationship_context:
        relationships = "; ".join(_extend_unique([], relationship_context))
        parts.append(f"relationships: {relationships}")
    return "\n".join(parts)


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        a * b for a, b in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


def _extend_unique(
    values: list[str],
    additions: Sequence[str],
) -> list[str]:
    merged = list(values)
    for addition in additions:
        if addition not in merged:
            merged.append(addition)
    return merged
