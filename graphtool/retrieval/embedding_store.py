import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class ChunkEmbeddingRecord(BaseModel):
    chunk_id: str
    embedding_model: str
    embedding_input_hash: str
    vector: list[float]


class ChunkEmbeddingStore(Protocol):
    def load(self) -> dict[str, ChunkEmbeddingRecord]:
        ...

    def save(self, records: Mapping[str, ChunkEmbeddingRecord]) -> None:
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        ...


class JsonChunkEmbeddingStore:
    """Filesystem-backed embedding cache for retrieval chunks."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, records: Mapping[str, ChunkEmbeddingRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [
                record.model_dump()
                for record in sorted(records.values(), key=lambda item: item.chunk_id)
            ]
        }
        self._path.write_text(json.dumps(data, indent=2))

    def load(self) -> dict[str, ChunkEmbeddingRecord]:
        if not self._path.exists():
            return {}

        data = json.loads(self._path.read_text())
        return {
            record.chunk_id: record
            for record in (
                ChunkEmbeddingRecord.model_validate(item)
                for item in data.get("records", [])
            )
        }

    def exists(self) -> bool:
        return self._path.exists()

    def delete(self, chunk_ids: list[str]) -> None:
        if not self._path.exists():
            return
        deleted = set(chunk_ids)
        records = {
            chunk_id: record
            for chunk_id, record in self.load().items()
            if chunk_id not in deleted
        }
        self.save(records)


def chunk_embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
