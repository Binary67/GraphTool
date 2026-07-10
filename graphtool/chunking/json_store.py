import json
from pathlib import Path

from graphtool.chunking.types import Chunk
from graphtool.source import source_key


class JsonChunkStore:
    """Filesystem-backed chunk store using JSON files."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(self, source: str, chunks: list[Chunk]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._path_for(source)
        data = [chunk.model_dump(mode="json") for chunk in chunks]
        path.write_text(json.dumps(data, indent=2))

    def load(self, source: str) -> list[Chunk]:
        path = self._path_for(source)
        data = json.loads(path.read_text())
        return [Chunk.model_validate(item) for item in data]

    def load_by_ids(self, source: str, chunk_ids: list[str]) -> list[Chunk]:
        chunks_by_id = {chunk.id: chunk for chunk in self.load(source)}
        return [
            chunks_by_id[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in chunks_by_id
        ]

    def load_neighborhood(
        self,
        source: str,
        chunk_id: str,
    ) -> tuple[Chunk | None, Chunk, Chunk | None]:
        try:
            chunks = sorted(self.load(source), key=lambda chunk: chunk.index)
        except FileNotFoundError:
            chunks = []

        position = next(
            (
                index
                for index, chunk in enumerate(chunks)
                if chunk.id == chunk_id
            ),
            None,
        )
        if position is None:
            raise ValueError(
                f"Chunk {chunk_id!r} was not found in source {source!r}."
            )

        previous = chunks[position - 1] if position > 0 else None
        current = chunks[position]
        next_chunk = chunks[position + 1] if position + 1 < len(chunks) else None
        return previous, current, next_chunk

    def load_all(self) -> list[Chunk]:
        if not self._directory.exists():
            return []
        chunks = []
        for path in sorted(self._directory.glob("*.json")):
            data = json.loads(path.read_text())
            chunks.extend(Chunk.model_validate(item) for item in data)
        return chunks

    def delete(self, source: str) -> None:
        self._path_for(source).unlink(missing_ok=True)

    def _path_for(self, source: str) -> Path:
        return self._directory / f"{source_key(source)}.json"
