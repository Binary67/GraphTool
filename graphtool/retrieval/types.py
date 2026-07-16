from pydantic import BaseModel, Field

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


class GraphPathHit(BaseModel):
    score: float
    nodes: list[Node]
    edges: list[Edge]
    chunk_ids: list[str]


class RetrievalResult(BaseModel):
    query: str
    sources: list[str]
    chunks: list[ChunkHit]
    graph_paths: list[GraphPathHit] = Field(default_factory=list)
    context_text: str
