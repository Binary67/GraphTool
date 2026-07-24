"""Shared SQLite helpers for the GraphTool persistence layer.

All migrated stores share a single database file. Each store accepts either a
``sqlite3.Connection`` (so ``create_runtime`` can share one connection across
stores) or a path (opened as a private connection, convenient for tests).
"""

import sqlite3
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_idx ON chunks(source, idx);

CREATE TABLE IF NOT EXISTS kb_node_embeddings (
    node_id TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_input_hash TEXT NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_node_embeddings (
    source TEXT NOT NULL,
    node_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_input_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (source, node_id)
);
CREATE INDEX IF NOT EXISTS idx_graph_node_embeddings_source
    ON graph_node_embeddings(source);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_input_hash TEXT NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS taxonomy_suggestions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    suggested_type TEXT NOT NULL,
    normalized_suggested_type TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_label TEXT NOT NULL,
    current_type TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_taxonomy_suggestions_source
    ON taxonomy_suggestions(source);
"""


def configure_connection(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)


def open_database(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    configure_connection(conn)
    return conn


def as_connection(conn_or_path: sqlite3.Connection | str | Path) -> sqlite3.Connection:
    if isinstance(conn_or_path, sqlite3.Connection):
        configure_connection(conn_or_path)
        return conn_or_path
    return open_database(conn_or_path)


def encode_vector(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def decode_vector(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()
