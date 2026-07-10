from pydantic import BaseModel

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, Node


class ChunkRelationship(BaseModel):
    edge: Edge
    source_node: Node
    target_node: Node


class ChunkHit(BaseModel):
    chunk: Chunk
    score: float
    linked_nodes: list[Node]
    linked_relationships: list[ChunkRelationship]


class RetrievalResult(BaseModel):
    query: str
    sources: list[str]
    chunks: list[ChunkHit]
    context_text: str
