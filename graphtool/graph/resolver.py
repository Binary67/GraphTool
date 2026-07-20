from collections.abc import Sequence

from graphtool.graph.entity_matching import (
    EntityResolutionDecision,
    accepted_target_id,
    find_normalized_match,
    judge_same_entity,
)
from graphtool.graph.provenance import (
    add_node_provenance,
    canonicalize_node,
    merge_nodes,
)
from graphtool.graph.resolution_embeddings import (
    EmbeddingStore,
    ResolutionEmbeddings,
    embedding_input_hash,
    node_embedding_text,
)
from graphtool.graph.resolution_merge import (
    dedupe_remapped_edges,
    merge_edge_sets,
    next_node_id,
    remap_edge,
    same_node_contribution,
)
from graphtool.graph.types import GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient, LLMClient

DEFAULT_TOP_K = 10
DEFAULT_MERGE_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MIN_CANDIDATE_SIMILARITY = 0.80


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
        self._embeddings = ResolutionEmbeddings(
            embedding_client,
            embedding_store,
        )
        self._top_k = top_k
        self._merge_confidence_threshold = merge_confidence_threshold
        self._min_candidate_similarity = min_candidate_similarity

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
        self._embeddings.prepare(
            all_graphs,
            [node for graph in graphs for node in graph.nodes],
        )

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
                remap_edge(edge, node_id_map) for edge in existing.edges
            ]
        else:
            existing_edges = []

        for graph in graphs:
            candidate_ids = None if resolve_within_graph else set(canonical_by_id)
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
                node_id_map[node.id] = self._resolve_node(
                    node,
                    canonical_nodes,
                    canonical_by_id,
                    candidates,
                    graph.metadata,
                )

        edges = dedupe_remapped_edges(graphs, node_id_map)
        graph = KnowledgeGraph(
            nodes=canonical_nodes,
            edges=merge_edge_sets(existing_edges, edges),
        )
        self._embeddings.finalize(canonical_nodes)
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
        if existing is not None and same_node_contribution(
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

        normalized_match = find_normalized_match(node, candidates)
        if normalized_match is not None:
            self._merge_into(
                normalized_match,
                node,
                canonical_nodes,
                canonical_by_id,
                metadata=metadata,
            )
            return normalized_match.id

        embedding_candidates = self._embeddings.candidates(
            node,
            candidates,
            min_similarity=self._min_candidate_similarity,
            top_k=self._top_k,
        )
        if embedding_candidates:
            decision = judge_same_entity(
                self._llm,
                node,
                embedding_candidates,
            )
            target_id = accepted_target_id(
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
                update={"id": next_node_id(canonical.id, canonical_by_id)}
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
        self._embeddings.invalidate(merged.id)
