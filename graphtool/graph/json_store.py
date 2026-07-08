import json
from pathlib import Path

from graphtool.graph.types import KnowledgeGraph
from graphtool.source import source_key


class JsonGraphStore:
    """Filesystem-backed knowledge graph store using JSON files."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(self, graph: KnowledgeGraph) -> None:
        if graph.metadata is None:
            raise ValueError("Cannot save graph without metadata.source.")
        self._directory.mkdir(parents=True, exist_ok=True)
        name = graph.metadata.source
        path = self._path_for(name)
        path.write_text(graph.model_dump_json(indent=2))

    def load(self, name: str) -> KnowledgeGraph:
        path = self._path_for(name)
        data = json.loads(path.read_text())
        return KnowledgeGraph.model_validate(data)

    def exists(self, source: str) -> bool:
        return self._path_for(source).exists()

    def load_all(self) -> list[KnowledgeGraph]:
        if not self._directory.exists():
            return []
        return [
            KnowledgeGraph.model_validate(json.loads(path.read_text()))
            for path in sorted(self._directory.glob("*.json"))
        ]

    def _path_for(self, name: str) -> Path:
        return self._directory / f"{source_key(name)}.json"
