from typing import Protocol

from graphtool.graph.types import KnowledgeGraph


class KnowledgeGraphStore(Protocol):
    """Common interface implemented by all knowledge graph stores."""

    def save(self, graph: KnowledgeGraph) -> None:
        """Persist a knowledge graph."""
        ...

    def load(self, name: str) -> KnowledgeGraph:
        """Load a knowledge graph by name."""
        ...

    def exists(self, source: str) -> bool:
        """Return whether a graph has been persisted for source."""
        ...

    def load_all(self) -> list[KnowledgeGraph]:
        """Load all persisted knowledge graphs."""
        ...
