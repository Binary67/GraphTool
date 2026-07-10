import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from graphtool.source import source_key


class NodeEmbeddingRecord(BaseModel):
    node_id: str
    embedding_model: str
    embedding_input_hash: str
    vector: list[float]


class JsonEmbeddingStore:
    """Filesystem-backed embedding cache for a single graph."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [
                record.model_dump()
                for record in sorted(records.values(), key=lambda item: item.node_id)
            ]
        }
        self._path.write_text(json.dumps(data, indent=2))

    def load(self) -> dict[str, NodeEmbeddingRecord]:
        if not self._path.exists():
            return {}

        data = json.loads(self._path.read_text())
        return {
            record.node_id: record
            for record in (
                NodeEmbeddingRecord.model_validate(item)
                for item in data.get("records", [])
            )
        }

    def exists(self) -> bool:
        return self._path.exists()


class JsonGraphEmbeddingStore:
    """Filesystem-backed embedding cache for per-document graphs."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(self, source: str, records: Mapping[str, NodeEmbeddingRecord]) -> None:
        JsonEmbeddingStore(self._path_for(source)).save(records)

    def load(self, source: str) -> dict[str, NodeEmbeddingRecord]:
        return JsonEmbeddingStore(self._path_for(source)).load()

    def exists(self, source: str) -> bool:
        return self._path_for(source).exists()

    def delete(self, source: str) -> None:
        self._path_for(source).unlink(missing_ok=True)

    def _path_for(self, source: str) -> Path:
        return self._directory / f"{source_key(source)}.json"
