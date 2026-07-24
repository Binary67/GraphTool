import sqlite3

from graphtool.retrieval.embedding_store import (
    ChunkEmbeddingRecord,
    SqliteChunkEmbeddingStore,
)
from graphtool.storage import open_database


def test_delete_batches_ids_beyond_the_sql_variable_limit(tmp_path):
    conn = open_database(tmp_path / "test.db")
    store = SqliteChunkEmbeddingStore(conn)
    records = {
        f"c{i}": ChunkEmbeddingRecord(
            chunk_id=f"c{i}",
            embedding_model="m",
            embedding_input_hash="h",
            vector=[0.0],
        )
        for i in range(1201)
    }
    store.save(records)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 500)

    store.delete(list(records))

    assert store.load() == {}
