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