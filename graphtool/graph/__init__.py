"""Knowledge graph generation and storage."""

from graphtool.graph.base import KnowledgeGraphStore
from graphtool.graph.embedding_store import (
    NodeEmbeddingRecord,
    SqliteEmbeddingStore,
    SqliteGraphEmbeddingStore,
)
from graphtool.graph.extraction_store import JsonChunkExtractionStore
from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.generator import generate_knowledge_graph
from graphtool.graph.sqlite_store import (
    SqliteGraphStore,
    SqliteKnowledgeBaseStore,
)
from graphtool.graph.provenance import filter_knowledge_graph_by_source
from graphtool.graph.resolver import SemanticEntityResolver
from graphtool.graph.taxonomy import (
    CANONICAL_NODE_TYPES,
    JsonNodeTypeRegistryStore,
    JsonTaxonomyPromotionAuditStore,
    NodeTypeRegistry,
    SqliteTaxonomySuggestionStore,
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
from graphtool.graph.types import (
    Edge,
    EdgeProvenance,
    GraphMetadata,
    KnowledgeGraph,
    Node,
    NodeProvenance,
)

__all__ = [
    "CANONICAL_NODE_TYPES",
    "Edge",
    "EdgeProvenance",
    "GraphMetadata",
    "JsonChunkExtractionStore",
    "SqliteEmbeddingStore",
    "SqliteGraphEmbeddingStore",
    "SqliteGraphStore",
    "SqliteKnowledgeBaseStore",
    "JsonNodeTypeRegistryStore",
    "JsonTaxonomyPromotionAuditStore",
    "SqliteTaxonomySuggestionStore",
    "KnowledgeGraph",
    "KnowledgeGraphStore",
    "NodeEmbeddingRecord",
    "NodeTypeRegistry",
    "Node",
    "NodeProvenance",
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
    "filter_knowledge_graph_by_source",
    "generate_knowledge_graph",
    "migrate_promoted_types",
    "normalize_type_name",
    "promote_suggestions",
]
