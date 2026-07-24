import json
import sqlite3
from pathlib import Path

from graphtool.chunking.types import Chunk
from graphtool.storage import as_connection

_SELECT_CHUNKS = (
    "SELECT id, source, idx, text, heading_path, page_start, page_end "
    "FROM chunks "
)


class SqliteChunkStore:
    """SQLite-backed chunk store."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def save(self, source: str, chunks: list[Chunk]) -> None:
        rows = [
            (
                chunk.id,
                source,
                chunk.index,
                chunk.text,
                json.dumps(chunk.heading_path),
                chunk.page_start,
                chunk.page_end,
            )
            for chunk in chunks
        ]
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            self._conn.executemany(
                "INSERT INTO chunks "
                "(id, source, idx, text, heading_path, page_start, page_end) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def load(self, source: str) -> list[Chunk]:
        rows = self._conn.execute(
            _SELECT_CHUNKS + "WHERE source = ? ORDER BY idx",
            (source,),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def load_by_ids(self, source: str, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            _SELECT_CHUNKS + f"WHERE source = ? AND id IN ({placeholders})",
            (source, *chunk_ids),
        ).fetchall()
        by_id = {row["id"]: self._row_to_chunk(row) for row in rows}
        return [
            by_id[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in by_id
        ]

    def load_neighborhood(
        self,
        source: str,
        chunk_id: str,
    ) -> tuple[Chunk | None, Chunk, Chunk | None]:
        current_row = self._conn.execute(
            _SELECT_CHUNKS + "WHERE source = ? AND id = ?",
            (source, chunk_id),
        ).fetchone()
        if current_row is None:
            raise ValueError(
                f"Chunk {chunk_id!r} was not found in source {source!r}."
            )
        previous_row = self._conn.execute(
            _SELECT_CHUNKS + "WHERE source = ? AND idx < ? ORDER BY idx DESC LIMIT 1",
            (source, current_row["idx"]),
        ).fetchone()
        next_row = self._conn.execute(
            _SELECT_CHUNKS + "WHERE source = ? AND idx > ? ORDER BY idx LIMIT 1",
            (source, current_row["idx"]),
        ).fetchone()
        return (
            self._row_to_chunk(previous_row) if previous_row is not None else None,
            self._row_to_chunk(current_row),
            self._row_to_chunk(next_row) if next_row is not None else None,
        )

    def load_all(self) -> list[Chunk]:
        rows = self._conn.execute(
            _SELECT_CHUNKS + "ORDER BY source, idx"
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def delete(self, source: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source=row["source"],
            index=row["idx"],
            text=row["text"],
            heading_path=json.loads(row["heading_path"]),
            page_start=row["page_start"],
            page_end=row["page_end"],
        )
