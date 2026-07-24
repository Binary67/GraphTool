import hashlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from graphtool.storage import as_connection, decode_vector, encode_vector, transaction

# Keeps DELETE ... IN (...) statements under SQLite's bound-variable limit.
_DELETE_BATCH_SIZE = 500


class ChunkEmbeddingRecord(BaseModel):
    chunk_id: str
    embedding_model: str
    embedding_input_hash: str
    vector: list[float]


class ChunkEmbeddingStore(Protocol):
    def load(self) -> dict[str, ChunkEmbeddingRecord]:
        ...

    def upsert(self, records: Mapping[str, ChunkEmbeddingRecord]) -> None:
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        ...


class SqliteChunkEmbeddingStore:
    """SQLite-backed embedding cache for retrieval chunks."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def upsert(self, records: Mapping[str, ChunkEmbeddingRecord]) -> None:
        with transaction(self._conn):
            self._conn.executemany(
                "INSERT INTO chunk_embeddings "
                "(chunk_id, embedding_model, embedding_input_hash, vector) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET "
                "embedding_model = excluded.embedding_model, "
                "embedding_input_hash = excluded.embedding_input_hash, "
                "vector = excluded.vector "
                "WHERE embedding_model <> excluded.embedding_model "
                "OR embedding_input_hash <> excluded.embedding_input_hash "
                "OR vector <> excluded.vector",
                [
                    (
                        record.chunk_id,
                        record.embedding_model,
                        record.embedding_input_hash,
                        encode_vector(record.vector),
                    )
                    for record in records.values()
                ],
            )

    def load(self) -> dict[str, ChunkEmbeddingRecord]:
        rows = self._conn.execute(
            "SELECT chunk_id, embedding_model, embedding_input_hash, vector "
            "FROM chunk_embeddings"
        ).fetchall()
        return {
            row["chunk_id"]: ChunkEmbeddingRecord(
                chunk_id=row["chunk_id"],
                embedding_model=row["embedding_model"],
                embedding_input_hash=row["embedding_input_hash"],
                vector=decode_vector(row["vector"]),
            )
            for row in rows
        }

    def exists(self) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS(SELECT 1 FROM chunk_embeddings LIMIT 1)"
        ).fetchone()
        return bool(row[0])

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with transaction(self._conn):
            for start in range(0, len(chunk_ids), _DELETE_BATCH_SIZE):
                batch = chunk_ids[start : start + _DELETE_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                self._conn.execute(
                    f"DELETE FROM chunk_embeddings "
                    f"WHERE chunk_id IN ({placeholders})",
                    batch,
                )


def chunk_embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
