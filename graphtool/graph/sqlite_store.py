import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from graphtool.graph.entity_matching import normalized_entity_name
from graphtool.graph.provenance import materialize_edge, materialize_node
from graphtool.graph.types import (
    Edge,
    EdgeProvenance,
    GraphMetadata,
    KnowledgeGraph,
    Node,
    NodeProvenance,
)
from graphtool.storage import as_connection, transaction

_BATCH_SIZE = 400


@dataclass(frozen=True)
class KnowledgeBaseDelta:
    upserted_nodes: list[Node]
    deleted_node_ids: set[str]
    upserted_edges: list[Edge]
    deleted_edge_ids: set[str]


class SqliteGraphStore:
    """SQLite-backed per-document graph store."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def save(self, graph: KnowledgeGraph) -> None:
        if graph.metadata is None:
            raise ValueError("Cannot save graph without metadata.source.")
        metadata = graph.metadata
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO graph_metadata "
                "(source, content_hash, model, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source) DO UPDATE SET "
                "content_hash = excluded.content_hash, "
                "model = excluded.model, "
                "created_at = excluded.created_at",
                (
                    metadata.source,
                    metadata.content_hash,
                    metadata.model,
                    metadata.created_at.isoformat(),
                ),
            )
            _sync_source_payloads(
                self._conn,
                "graph_nodes",
                "node_id",
                metadata.source,
                ((node.id, node.model_dump_json()) for node in graph.nodes),
            )
            _sync_source_payloads(
                self._conn,
                "graph_edges",
                "edge_id",
                metadata.source,
                ((edge.id, edge.model_dump_json()) for edge in graph.edges),
            )

    def load(self, source: str) -> KnowledgeGraph:
        metadata_row = self._conn.execute(
            "SELECT source, content_hash, model, created_at "
            "FROM graph_metadata WHERE source = ?",
            (source,),
        ).fetchone()
        if metadata_row is None:
            raise FileNotFoundError(f"Graph for {source!r} was not found.")
        nodes = [
            Node.model_validate_json(row["payload"])
            for row in self._conn.execute(
                "SELECT payload FROM graph_nodes "
                "WHERE source = ? ORDER BY rowid",
                (source,),
            )
        ]
        edges = [
            Edge.model_validate_json(row["payload"])
            for row in self._conn.execute(
                "SELECT payload FROM graph_edges "
                "WHERE source = ? ORDER BY rowid",
                (source,),
            )
        ]
        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            metadata=_metadata_from_row(metadata_row),
        )

    def load_all(self) -> list[KnowledgeGraph]:
        sources = [metadata.source for metadata in self.load_metadata()]
        return [self.load(source) for source in sources]

    def load_metadata(self) -> list[GraphMetadata]:
        return [
            _metadata_from_row(row)
            for row in self._conn.execute(
                "SELECT source, content_hash, model, created_at "
                "FROM graph_metadata ORDER BY source"
            )
        ]

    def transaction(self):
        return transaction(self._conn)

    def exists(self, source: str) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM graph_metadata WHERE source = ? LIMIT 1)",
            (source,),
        ).fetchone()
        return bool(row[0])

    def delete(self, source: str) -> None:
        with transaction(self._conn):
            self._conn.execute("DELETE FROM graph_nodes WHERE source = ?", (source,))
            self._conn.execute("DELETE FROM graph_edges WHERE source = ?", (source,))
            self._conn.execute(
                "DELETE FROM graph_metadata WHERE source = ?",
                (source,),
            )


class SqliteKnowledgeBaseStore:
    """Incremental SQLite-backed canonical knowledge graph store."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def replace_all(self, graph: KnowledgeGraph) -> None:
        node_rows = {
            node.id: node.model_copy(update={"provenance": []}).model_dump_json()
            for node in graph.nodes
        }
        alias_rows = {
            (node.id, normalized): alias
            for node in graph.nodes
            for alias in [node.label, *node.aliases]
            if (normalized := normalized_entity_name(alias))
        }
        node_provenance_rows = {
            (
                node.id,
                provenance.source,
                provenance.content_hash,
                provenance.node_id,
            ): provenance.model_dump_json()
            for node in graph.nodes
            for provenance in node.provenance
        }
        edge_rows = {
            edge.id: (
                edge.source,
                edge.target,
                edge.model_copy(update={"provenance": []}).model_dump_json(),
            )
            for edge in graph.edges
        }
        edge_provenance_rows = {
            (
                edge.id,
                provenance.source,
                provenance.content_hash,
                provenance.edge_id,
            ): provenance.model_dump_json()
            for edge in graph.edges
            for provenance in edge.provenance
        }

        with transaction(self._conn):
            self._conn.execute(
                "INSERT OR IGNORE INTO knowledge_base_state(singleton) VALUES (1)"
            )
            _sync_keyed_payloads(
                self._conn,
                "knowledge_base_nodes",
                "node_id",
                node_rows,
            )
            _sync_aliases(self._conn, alias_rows)
            _sync_provenance(
                self._conn,
                "knowledge_base_node_provenance",
                (
                    "canonical_node_id",
                    "source",
                    "content_hash",
                    "source_node_id",
                ),
                node_provenance_rows,
            )
            _sync_edges(self._conn, edge_rows)
            _sync_provenance(
                self._conn,
                "knowledge_base_edge_provenance",
                (
                    "canonical_edge_id",
                    "source",
                    "content_hash",
                    "source_edge_id",
                ),
                edge_provenance_rows,
            )

    def affected_ids(
        self,
        sources: Sequence[str],
    ) -> tuple[set[str], set[str]]:
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        for batch in _batches(sources):
            placeholders = ",".join("?" for _ in batch)
            node_ids.update(
                row["canonical_node_id"]
                for row in self._conn.execute(
                    "SELECT DISTINCT canonical_node_id "
                    "FROM knowledge_base_node_provenance "
                    f"WHERE source IN ({placeholders})",
                    batch,
                )
            )
            edge_ids.update(
                row["canonical_edge_id"]
                for row in self._conn.execute(
                    "SELECT DISTINCT canonical_edge_id "
                    "FROM knowledge_base_edge_provenance "
                    f"WHERE source IN ({placeholders})",
                    batch,
                )
            )
        for batch in _batches(sorted(node_ids)):
            placeholders = ",".join("?" for _ in batch)
            edge_ids.update(
                row["edge_id"]
                for row in self._conn.execute(
                    "SELECT edge_id FROM knowledge_base_edges "
                    f"WHERE source_node_id IN ({placeholders}) "
                    f"OR target_node_id IN ({placeholders})",
                    (*batch, *batch),
                )
            )
        return node_ids, edge_ids

    def apply_delta(self, delta: KnowledgeBaseDelta) -> None:
        node_ids = [node.id for node in delta.upserted_nodes]
        edge_ids = [edge.id for edge in delta.upserted_edges]
        with transaction(self._conn):
            self._delete_edges(delta.deleted_edge_ids)
            self._delete_nodes(delta.deleted_node_ids)
            self._conn.executemany(
                "INSERT INTO knowledge_base_nodes (node_id, payload) VALUES (?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET payload = excluded.payload "
                "WHERE payload <> excluded.payload",
                [
                    (
                        node.id,
                        node.model_copy(update={"provenance": []}).model_dump_json(),
                    )
                    for node in delta.upserted_nodes
                ],
            )
            self._conn.executemany(
                "DELETE FROM knowledge_base_node_aliases WHERE node_id = ?",
                [(node_id,) for node_id in node_ids],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_node_aliases "
                "(node_id, alias, normalized_alias) VALUES (?, ?, ?)",
                [
                    (node.id, alias, normalized)
                    for node in delta.upserted_nodes
                    for alias in [node.label, *node.aliases]
                    if (normalized := normalized_entity_name(alias))
                ],
            )
            self._conn.executemany(
                "DELETE FROM knowledge_base_node_provenance "
                "WHERE canonical_node_id = ?",
                [(node_id,) for node_id in node_ids],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_node_provenance "
                "(canonical_node_id, source, content_hash, source_node_id, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        node.id,
                        provenance.source,
                        provenance.content_hash,
                        provenance.node_id,
                        provenance.model_dump_json(),
                    )
                    for node in delta.upserted_nodes
                    for provenance in node.provenance
                ],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_edges "
                "(edge_id, source_node_id, target_node_id, payload) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(edge_id) DO UPDATE SET "
                "source_node_id = excluded.source_node_id, "
                "target_node_id = excluded.target_node_id, "
                "payload = excluded.payload "
                "WHERE source_node_id <> excluded.source_node_id "
                "OR target_node_id <> excluded.target_node_id "
                "OR payload <> excluded.payload",
                [
                    (
                        edge.id,
                        edge.source,
                        edge.target,
                        edge.model_copy(
                            update={"provenance": []}
                        ).model_dump_json(),
                    )
                    for edge in delta.upserted_edges
                ],
            )
            self._conn.executemany(
                "DELETE FROM knowledge_base_edge_provenance "
                "WHERE canonical_edge_id = ?",
                [(edge_id,) for edge_id in edge_ids],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_edge_provenance "
                "(canonical_edge_id, source, content_hash, source_edge_id, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        edge.id,
                        provenance.source,
                        provenance.content_hash,
                        provenance.edge_id,
                        provenance.model_dump_json(),
                    )
                    for edge in delta.upserted_edges
                    for provenance in edge.provenance
                ],
            )

    def _delete_nodes(self, node_ids: set[str]) -> None:
        rows = [(node_id,) for node_id in node_ids]
        self._conn.executemany(
            "DELETE FROM knowledge_base_node_aliases WHERE node_id = ?",
            rows,
        )
        self._conn.executemany(
            "DELETE FROM knowledge_base_node_provenance "
            "WHERE canonical_node_id = ?",
            rows,
        )
        self._conn.executemany(
            "DELETE FROM knowledge_base_nodes WHERE node_id = ?",
            rows,
        )

    def _delete_edges(self, edge_ids: set[str]) -> None:
        rows = [(edge_id,) for edge_id in edge_ids]
        self._conn.executemany(
            "DELETE FROM knowledge_base_edge_provenance "
            "WHERE canonical_edge_id = ?",
            rows,
        )
        self._conn.executemany(
            "DELETE FROM knowledge_base_edges WHERE edge_id = ?",
            rows,
        )

    def load(self) -> KnowledgeGraph:
        if not self.exists():
            raise FileNotFoundError("Knowledge base was not found.")
        return self._load_excluding_sources(())

    def load_excluding_sources(self, sources: Sequence[str]) -> KnowledgeGraph:
        if not self.exists():
            raise FileNotFoundError("Knowledge base was not found.")
        return self._load_excluding_sources(sources)

    def _load_excluding_sources(
        self,
        sources: Sequence[str],
    ) -> KnowledgeGraph:
        all_node_ids_with_provenance = {
            row["canonical_node_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT canonical_node_id "
                "FROM knowledge_base_node_provenance"
            )
        }
        node_provenance = _load_node_provenance(
            self._conn,
            excluded_sources=sources,
        )
        nodes = []
        for row in self._conn.execute(
            "SELECT node_id, payload FROM knowledge_base_nodes ORDER BY rowid"
        ):
            base = Node.model_validate_json(row["payload"])
            provenance = node_provenance.get(row["node_id"], [])
            if (
                row["node_id"] in all_node_ids_with_provenance
                and not provenance
            ):
                continue
            nodes.append(
                (
                    materialize_node(base.id, provenance)
                    if sources
                    else base.model_copy(update={"provenance": provenance})
                )
                if provenance
                else base
            )

        node_ids = {node.id for node in nodes}
        all_edge_ids_with_provenance = {
            row["canonical_edge_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT canonical_edge_id "
                "FROM knowledge_base_edge_provenance"
            )
        }
        edge_provenance = _load_edge_provenance(
            self._conn,
            excluded_sources=sources,
        )
        edges = []
        for row in self._conn.execute(
            "SELECT edge_id, payload FROM knowledge_base_edges ORDER BY rowid"
        ):
            base = Edge.model_validate_json(row["payload"])
            provenance = edge_provenance.get(row["edge_id"], [])
            if base.source not in node_ids or base.target not in node_ids:
                continue
            if (
                row["edge_id"] in all_edge_ids_with_provenance
                and not provenance
            ):
                continue
            edges.append(
                (
                    materialize_edge(
                        base.id,
                        base.source,
                        base.target,
                        provenance,
                    )
                    if sources
                    else base.model_copy(update={"provenance": provenance})
                )
                if provenance
                else base
            )
        return KnowledgeGraph(nodes=nodes, edges=edges)

    def exists(self) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM knowledge_base_state WHERE singleton = 1 LIMIT 1)"
        ).fetchone()
        return bool(row[0])

def _metadata_from_row(row: sqlite3.Row) -> GraphMetadata:
    return GraphMetadata.model_validate(
        {
            "source": row["source"],
            "content_hash": row["content_hash"],
            "model": row["model"],
            "created_at": row["created_at"],
        }
    )


def _sync_source_payloads(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    source: str,
    rows: Iterable[tuple[str, str]],
) -> None:
    desired = dict(rows)
    existing = {
        row[id_column]: row["payload"]
        for row in conn.execute(
            f"SELECT {id_column}, payload FROM {table} WHERE source = ?",
            (source,),
        )
    }
    removed = set(existing) - set(desired)
    conn.executemany(
        f"DELETE FROM {table} WHERE source = ? AND {id_column} = ?",
        [(source, item_id) for item_id in removed],
    )
    conn.executemany(
        f"INSERT INTO {table} (source, {id_column}, payload) VALUES (?, ?, ?) "
        f"ON CONFLICT(source, {id_column}) DO UPDATE SET payload = excluded.payload "
        f"WHERE payload <> excluded.payload",
        [
            (source, item_id, payload)
            for item_id, payload in desired.items()
            if existing.get(item_id) != payload
        ],
    )


def _sync_keyed_payloads(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    desired: dict[str, str],
) -> None:
    existing = {
        row[id_column]: row["payload"]
        for row in conn.execute(f"SELECT {id_column}, payload FROM {table}")
    }
    conn.executemany(
        f"DELETE FROM {table} WHERE {id_column} = ?",
        [(item_id,) for item_id in set(existing) - set(desired)],
    )
    conn.executemany(
        f"INSERT INTO {table} ({id_column}, payload) VALUES (?, ?) "
        f"ON CONFLICT({id_column}) DO UPDATE SET payload = excluded.payload "
        f"WHERE payload <> excluded.payload",
        [
            (item_id, payload)
            for item_id, payload in desired.items()
            if existing.get(item_id) != payload
        ],
    )


def _sync_aliases(
    conn: sqlite3.Connection,
    desired: dict[tuple[str, str], str],
) -> None:
    existing = {
        (row["node_id"], row["normalized_alias"]): row["alias"]
        for row in conn.execute(
            "SELECT node_id, normalized_alias, alias "
            "FROM knowledge_base_node_aliases"
        )
    }
    conn.executemany(
        "DELETE FROM knowledge_base_node_aliases "
        "WHERE node_id = ? AND normalized_alias = ?",
        list(set(existing) - set(desired)),
    )
    conn.executemany(
        "INSERT INTO knowledge_base_node_aliases "
        "(node_id, normalized_alias, alias) VALUES (?, ?, ?) "
        "ON CONFLICT(node_id, normalized_alias) DO UPDATE SET alias = excluded.alias "
        "WHERE alias <> excluded.alias",
        [
            (node_id, normalized, alias)
            for (node_id, normalized), alias in desired.items()
            if existing.get((node_id, normalized)) != alias
        ],
    )


def _sync_edges(
    conn: sqlite3.Connection,
    desired: dict[str, tuple[str, str, str]],
) -> None:
    existing = {
        row["edge_id"]: (
            row["source_node_id"],
            row["target_node_id"],
            row["payload"],
        )
        for row in conn.execute(
            "SELECT edge_id, source_node_id, target_node_id, payload "
            "FROM knowledge_base_edges"
        )
    }
    conn.executemany(
        "DELETE FROM knowledge_base_edges WHERE edge_id = ?",
        [(edge_id,) for edge_id in set(existing) - set(desired)],
    )
    conn.executemany(
        "INSERT INTO knowledge_base_edges "
        "(edge_id, source_node_id, target_node_id, payload) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(edge_id) DO UPDATE SET "
        "source_node_id = excluded.source_node_id, "
        "target_node_id = excluded.target_node_id, "
        "payload = excluded.payload "
        "WHERE source_node_id <> excluded.source_node_id "
        "OR target_node_id <> excluded.target_node_id "
        "OR payload <> excluded.payload",
        [
            (edge_id, source, target, payload)
            for edge_id, (source, target, payload) in desired.items()
            if existing.get(edge_id) != (source, target, payload)
        ],
    )


def _sync_provenance(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, str, str, str],
    desired: dict[tuple[str, str, str, str], str],
) -> None:
    column_list = ", ".join(columns)
    existing = {
        tuple(row[column] for column in columns): row["payload"]
        for row in conn.execute(f"SELECT {column_list}, payload FROM {table}")
    }
    where = " AND ".join(f"{column} = ?" for column in columns)
    conn.executemany(
        f"DELETE FROM {table} WHERE {where}",
        list(set(existing) - set(desired)),
    )
    conflict = ", ".join(columns)
    placeholders = ", ".join("?" for _ in range(5))
    conn.executemany(
        f"INSERT INTO {table} ({column_list}, payload) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET payload = excluded.payload "
        f"WHERE payload <> excluded.payload",
        [
            (*key, payload)
            for key, payload in desired.items()
            if existing.get(key) != payload
        ],
    )


def _load_node_provenance(
    conn: sqlite3.Connection,
    node_id: str | None = None,
    *,
    excluded_sources: Sequence[str] = (),
) -> dict[str, list[NodeProvenance]]:
    query = (
        "SELECT canonical_node_id, payload "
        "FROM knowledge_base_node_provenance"
    )
    parameters: tuple[str, ...] = ()
    if node_id is not None:
        query += " WHERE canonical_node_id = ?"
        parameters = (node_id,)
    if excluded_sources:
        conjunction = " AND " if parameters else " WHERE "
        placeholders = ",".join("?" for _ in excluded_sources)
        query += f"{conjunction}source NOT IN ({placeholders})"
        parameters = (*parameters, *excluded_sources)
    query += " ORDER BY rowid"
    result: dict[str, list[NodeProvenance]] = {}
    for row in conn.execute(query, parameters):
        result.setdefault(row["canonical_node_id"], []).append(
            NodeProvenance.model_validate_json(row["payload"])
        )
    return result


def _batches(values: Sequence[str]) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), _BATCH_SIZE):
        yield tuple(values[start : start + _BATCH_SIZE])


def _load_edge_provenance(
    conn: sqlite3.Connection,
    edge_id: str | None = None,
    *,
    excluded_sources: Sequence[str] = (),
) -> dict[str, list[EdgeProvenance]]:
    query = (
        "SELECT canonical_edge_id, payload "
        "FROM knowledge_base_edge_provenance"
    )
    parameters: tuple[str, ...] = ()
    if edge_id is not None:
        query += " WHERE canonical_edge_id = ?"
        parameters = (edge_id,)
    if excluded_sources:
        conjunction = " AND " if parameters else " WHERE "
        placeholders = ",".join("?" for _ in excluded_sources)
        query += f"{conjunction}source NOT IN ({placeholders})"
        parameters = (*parameters, *excluded_sources)
    query += " ORDER BY rowid"
    result: dict[str, list[EdgeProvenance]] = {}
    for row in conn.execute(query, parameters):
        result.setdefault(row["canonical_edge_id"], []).append(
            EdgeProvenance.model_validate_json(row["payload"])
        )
    return result
