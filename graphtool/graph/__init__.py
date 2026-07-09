"""Knowledge graph generation and storage."""

from graphtool.graph.base import KnowledgeGraphStore
from graphtool.graph.embedding_store import (
    JsonEmbeddingStore,
    JsonGraphEmbeddingStore,
    NodeEmbeddingRecord,
)
from graphtool.graph.generator import combine_knowledge_graphs, generate_knowledge_graph
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.resolver import SemanticEntityResolver
from graphtool.graph.taxonomy import (
    CANONICAL_NODE_TYPES,
    JsonNodeTypeRegistryStore,
    JsonTaxonomyPromotionAuditStore,
    JsonTaxonomySuggestionStore,
    NodeTypeRegistry,
    TaxonomyEvolutionResult,
    TaxonomyPromotionRecord,
    TaxonomySuggestionAggregate,
    TaxonomySuggestionRecord,
    UNCLASSIFIED_NODE_TYPE,
    aggregate_suggestions,
    canonical_node_type_text,
    default_node_type_registry,
    evolve_taxonomy,
    migrate_promoted_types,
    normalize_type_name,
    promote_suggestions,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node

__all__ = [
    "CANONICAL_NODE_TYPES",
    "Edge",
    "GraphMetadata",
    "JsonEmbeddingStore",
    "JsonGraphEmbeddingStore",
    "JsonGraphStore",
    "JsonKnowledgeBaseStore",
    "JsonNodeTypeRegistryStore",
    "JsonTaxonomyPromotionAuditStore",
    "JsonTaxonomySuggestionStore",
    "KnowledgeGraph",
    "KnowledgeGraphStore",
    "NodeEmbeddingRecord",
    "NodeTypeRegistry",
    "Node",
    "SemanticEntityResolver",
    "TaxonomyEvolutionResult",
    "TaxonomyPromotionRecord",
    "TaxonomySuggestionAggregate",
    "TaxonomySuggestionRecord",
    "UNCLASSIFIED_NODE_TYPE",
    "aggregate_suggestions",
    "canonical_node_type_text",
    "combine_knowledge_graphs",
    "default_node_type_registry",
    "evolve_taxonomy",
    "generate_knowledge_graph",
    "migrate_promoted_types",
    "normalize_type_name",
    "promote_suggestions",
]
