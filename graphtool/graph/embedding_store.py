import sqlite3
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from graphtool.storage import as_connection, decode_vector, encode_vector


class NodeEmbeddingRecord(BaseModel):
    node_id: str
    embedding_model: str
    embedding_input_hash: str
    vector: list[float]


def _row_to_node_record(row: sqlite3.Row) -> NodeEmbeddingRecord:
    return NodeEmbeddingRecord(
        node_id=row["node_id"],
        embedding_model=row["embedding_model"],
        embedding_input_hash=row["embedding_input_hash"],
        vector=decode_vector(row["vector"]),
    )


class SqliteEmbeddingStore:
    """SQLite-backed embedding cache for a single graph (the knowledge base)."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def save(self, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        rows = [
            (
                record.node_id,
                record.embedding_model,
                record.embedding_input_hash,
                encode_vector(record.vector),
            )
            for record in records.values()
        ]
        with self._conn:
            self._conn.execute("DELETE FROM kb_node_embeddings")
            self._conn.executemany(
                "INSERT INTO kb_node_embeddings "
                "(node_id, embedding_model, embedding_input_hash, vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def load(self) -> dict[str, NodeEmbeddingRecord]:
        rows = self._conn.execute(
            "SELECT node_id, embedding_model, embedding_input_hash, vector "
            "FROM kb_node_embeddings"
        ).fetchall()
        return {row["node_id"]: _row_to_node_record(row) for row in rows}

    def exists(self) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS(SELECT 1 FROM kb_node_embeddings LIMIT 1)"
        ).fetchone()
        return bool(row[0])


class SqliteGraphEmbeddingStore:
    """SQLite-backed embedding cache for per-document graphs."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def save(self, source: str, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        rows = [
            (
                source,
                record.node_id,
                record.embedding_model,
                record.embedding_input_hash,
                encode_vector(record.vector),
            )
            for record in records.values()
        ]
        with self._conn:
            self._conn.execute(
                "DELETE FROM graph_node_embeddings WHERE source = ?", (source,)
            )
            self._conn.executemany(
                "INSERT INTO graph_node_embeddings "
                "(source, node_id, embedding_model, embedding_input_hash, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def load(self, source: str) -> dict[str, NodeEmbeddingRecord]:
        rows = self._conn.execute(
            "SELECT node_id, embedding_model, embedding_input_hash, vector "
            "FROM graph_node_embeddings WHERE source = ?",
            (source,),
        ).fetchall()
        return {row["node_id"]: _row_to_node_record(row) for row in rows}

    def exists(self, source: str) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM graph_node_embeddings WHERE source = ? LIMIT 1)",
            (source,),
        ).fetchone()
        return bool(row[0])

    def delete(self, source: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM graph_node_embeddings WHERE source = ?", (source,)
            )

    def for_source(self, source: str) -> "_SourceScopedEmbeddingStore":
        return _SourceScopedEmbeddingStore(self, source)


class _SourceScopedEmbeddingStore:
    """Adapts one source of a graph store to the single-graph store interface."""

    def __init__(self, store: SqliteGraphEmbeddingStore, source: str) -> None:
        self._store = store
        self._source = source

    def load(self) -> dict[str, NodeEmbeddingRecord]:
        return self._store.load(self._source)

    def save(self, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        self._store.save(self._source, records)
