import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtool.graph.types import KnowledgeGraph

UNCLASSIFIED_NODE_TYPE = "unclassified"

CanonicalNodeType = Literal[
    "concept",
    "feature",
    "capability",
    "tool",
    "integration",
    "product",
    "service",
    "organization",
    "person",
    "document",
    "repository",
    "package",
    "plugin",
    "model",
    "process",
    "system",
    "agent",
    "environment",
    "event_trigger",
    "resource",
    "unclassified",
]

CANONICAL_NODE_TYPES: tuple[str, ...] = (
    "concept",
    "feature",
    "capability",
    "tool",
    "integration",
    "product",
    "service",
    "organization",
    "person",
    "document",
    "repository",
    "package",
    "plugin",
    "model",
    "process",
    "system",
    "agent",
    "environment",
    "event_trigger",
    "resource",
    "unclassified",
)

DEFAULT_TAXONOMY_VERSION = 1
DEFAULT_PROMOTION_MIN_NODES = 5
DEFAULT_PROMOTION_MIN_SOURCES = 2


class TaxonomySuggestionStore(Protocol):
    def append_many(self, records: Sequence["TaxonomySuggestionRecord"]) -> None:
        ...


class NodeTypeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""


class NodeTypeRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = DEFAULT_TAXONOMY_VERSION
    types: dict[str, NodeTypeDefinition] = Field(
        default_factory=lambda: {
            type_name: NodeTypeDefinition()
            for type_name in CANONICAL_NODE_TYPES
        }
    )

    def with_promoted_types(
        self,
        promoted_types: Sequence[str],
    ) -> "NodeTypeRegistry":
        types = dict(self.types)
        for type_name in promoted_types:
            normalized = normalize_type_name(type_name)
            if normalized in types:
                continue
            types[normalized] = NodeTypeDefinition(
                description=f"Promoted from suggested type {normalized}."
            )
        return self.model_copy(update={"version": self.version + 1, "types": types})


class TaxonomySuggestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggested_type: str
    normalized_suggested_type: str
    node_id: str
    node_label: str
    current_type: str
    source: str
    chunk_id: str
    model: str | None = None
    created_at: datetime


class TaxonomySuggestionAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_suggested_type: str
    suggested_types: list[str]
    node_count: int
    source_count: int
    node_ids: list[str]
    sources: list[str]


class TaxonomyPromotionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["promote_type"] = "promote_type"
    type: str
    matched_suggestions: list[str]
    affected_nodes: list[str]
    reason: str
    created_at: datetime


class TaxonomyEvolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry: NodeTypeRegistry
    graphs: list[KnowledgeGraph]
    promotions: list[TaxonomyPromotionRecord]


class JsonNodeTypeRegistryStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, registry: NodeTypeRegistry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(registry.model_dump_json(indent=2))

    def load(self) -> NodeTypeRegistry:
        data = json.loads(self._path.read_text())
        return NodeTypeRegistry.model_validate(data)

    def load_or_default(self) -> NodeTypeRegistry:
        if not self.exists():
            return default_node_type_registry()
        return self.load()

    def exists(self) -> bool:
        return self._path.exists()


class JsonTaxonomySuggestionStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append_many(self, records: Sequence[TaxonomySuggestionRecord]) -> None:
        if not records:
            return
        existing = self.load()
        self.save([*existing, *records])

    def save(self, records: Sequence[TaxonomySuggestionRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                indent=2,
                sort_keys=True,
            )
        )

    def load(self) -> list[TaxonomySuggestionRecord]:
        if not self.exists():
            return []
        data = json.loads(self._path.read_text())
        return [TaxonomySuggestionRecord.model_validate(item) for item in data]

    def exists(self) -> bool:
        return self._path.exists()


class JsonTaxonomyPromotionAuditStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append_many(self, records: Sequence[TaxonomyPromotionRecord]) -> None:
        if not records:
            return
        existing = self.load()
        self.save([*existing, *records])

    def save(self, records: Sequence[TaxonomyPromotionRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                indent=2,
                sort_keys=True,
            )
        )

    def load(self) -> list[TaxonomyPromotionRecord]:
        if not self.exists():
            return []
        data = json.loads(self._path.read_text())
        return [TaxonomyPromotionRecord.model_validate(item) for item in data]

    def exists(self) -> bool:
        return self._path.exists()


def default_node_type_registry() -> NodeTypeRegistry:
    return NodeTypeRegistry()


def canonical_node_type_text() -> str:
    return ", ".join(CANONICAL_NODE_TYPES)


def make_taxonomy_suggestion_records(
    *,
    nodes: Sequence[object],
    source: str,
    chunk_id: str,
    model: str | None = None,
    created_at: datetime | None = None,
) -> list[TaxonomySuggestionRecord]:
    timestamp = created_at or datetime.now(timezone.utc)
    records = []
    for node in nodes:
        suggested_type = getattr(node, "suggested_type", None)
        if suggested_type is None:
            continue
        normalized = normalize_type_name(suggested_type)
        if not normalized:
            continue
        records.append(
            TaxonomySuggestionRecord(
                suggested_type=suggested_type,
                normalized_suggested_type=normalized,
                node_id=getattr(node, "id"),
                node_label=getattr(node, "label"),
                current_type=getattr(node, "type"),
                source=source,
                chunk_id=chunk_id,
                model=model,
                created_at=timestamp,
            )
        )
    return records


def aggregate_suggestions(
    records: Sequence[TaxonomySuggestionRecord],
) -> list[TaxonomySuggestionAggregate]:
    grouped: dict[str, list[TaxonomySuggestionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.normalized_suggested_type].append(record)

    aggregates = []
    for normalized, group in sorted(grouped.items()):
        node_keys = _unique(
            _node_key(record.source, record.node_id)
            for record in group
        )
        aggregates.append(
            TaxonomySuggestionAggregate(
                normalized_suggested_type=normalized,
                suggested_types=_unique(record.suggested_type for record in group),
                node_count=len(node_keys),
                source_count=len({record.source for record in group}),
                node_ids=node_keys,
                sources=_unique(record.source for record in group),
            )
        )
    return aggregates


def evolve_taxonomy(
    registry_store: JsonNodeTypeRegistryStore,
    suggestion_store: JsonTaxonomySuggestionStore,
    audit_store: JsonTaxonomyPromotionAuditStore,
    graphs: Sequence[KnowledgeGraph],
    *,
    min_nodes: int = DEFAULT_PROMOTION_MIN_NODES,
    min_sources: int = DEFAULT_PROMOTION_MIN_SOURCES,
) -> TaxonomyEvolutionResult:
    result = promote_suggestions(
        registry_store.load_or_default(),
        suggestion_store.load(),
        graphs,
        min_nodes=min_nodes,
        min_sources=min_sources,
    )
    registry_store.save(result.registry)
    audit_store.append_many(result.promotions)
    return result


def promote_suggestions(
    registry: NodeTypeRegistry,
    records: Sequence[TaxonomySuggestionRecord],
    graphs: Sequence[KnowledgeGraph],
    *,
    min_nodes: int = DEFAULT_PROMOTION_MIN_NODES,
    min_sources: int = DEFAULT_PROMOTION_MIN_SOURCES,
    created_at: datetime | None = None,
) -> TaxonomyEvolutionResult:
    timestamp = created_at or datetime.now(timezone.utc)
    existing_types = {normalize_type_name(type_name) for type_name in registry.types}
    promoted_types = []
    promotions = []

    for aggregate in aggregate_suggestions(records):
        promoted_type = aggregate.normalized_suggested_type
        if promoted_type in existing_types:
            continue
        if aggregate.node_count < min_nodes:
            continue
        if aggregate.source_count < min_sources:
            continue

        promoted_types.append(promoted_type)
        existing_types.add(promoted_type)
        promotions.append(
            TaxonomyPromotionRecord(
                type=promoted_type,
                matched_suggestions=aggregate.suggested_types,
                affected_nodes=[],
                reason=(
                    f"Appeared in {aggregate.node_count} nodes across "
                    f"{aggregate.source_count} source documents."
                ),
                created_at=timestamp,
            )
        )

    promoted_lookup = set(promoted_types)
    migrated_graphs = [
        migrate_promoted_types(graph, promoted_lookup)
        for graph in graphs
    ]
    affected_by_type = _affected_nodes_by_type(graphs, migrated_graphs, promoted_lookup)
    promotions = [
        promotion.model_copy(
            update={"affected_nodes": affected_by_type.get(promotion.type, [])}
        )
        for promotion in promotions
    ]

    registry = (
        registry.with_promoted_types(promoted_types)
        if promoted_types
        else registry
    )
    return TaxonomyEvolutionResult(
        registry=registry,
        graphs=migrated_graphs,
        promotions=promotions,
    )


def migrate_promoted_types(
    graph: KnowledgeGraph,
    promoted_types: set[str],
) -> KnowledgeGraph:
    nodes = []
    for node in graph.nodes:
        suggested_type = node.suggested_type
        should_migrate = (
            normalize_type_name(node.type) == UNCLASSIFIED_NODE_TYPE
            and suggested_type is not None
            and normalize_type_name(suggested_type) in promoted_types
        )
        if not should_migrate:
            nodes.append(node)
            continue

        nodes.append(
            node.model_copy(
                update={"type": normalize_type_name(suggested_type)}
            )
        )
    return graph.model_copy(update={"nodes": nodes})


def normalize_type_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold())
    return "_".join(part for part in normalized.split("_") if part)


def _affected_nodes_by_type(
    original_graphs: Sequence[KnowledgeGraph],
    migrated_graphs: Sequence[KnowledgeGraph],
    promoted_types: set[str],
) -> dict[str, list[str]]:
    affected: dict[str, list[str]] = {type_name: [] for type_name in promoted_types}
    for original, migrated in zip(original_graphs, migrated_graphs, strict=True):
        source = original.metadata.source if original.metadata is not None else ""
        for original_node, migrated_node in zip(
            original.nodes,
            migrated.nodes,
            strict=True,
        ):
            if original_node.type == migrated_node.type:
                continue
            promoted_type = normalize_type_name(migrated_node.type)
            if promoted_type in affected:
                affected[promoted_type].append(
                    _node_key(source, migrated_node.id)
                    if source
                    else migrated_node.id
                )
    return {
        type_name: _unique(nodes)
        for type_name, nodes in affected.items()
    }


def _node_key(source: str, node_id: str) -> str:
    return f"{source}:{node_id}"


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
