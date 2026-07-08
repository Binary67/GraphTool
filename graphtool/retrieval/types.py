from pydantic import BaseModel

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, Node


class NodeHit(BaseModel):
    node: Node
    score: float
    matched_text: str | None = None


class RelationshipHit(BaseModel):
    edge: Edge
    source_node: Node
    target_node: Node
    score: float
    chunk_ids: list[str]


class ChunkHit(BaseModel):
    chunk: Chunk
    score: float
    linked_node_ids: list[str]
    linked_edge_ids: list[str]


class RetrievalResult(BaseModel):
    query: str
    source: str
    node_hits: list[NodeHit]
    relationship_hits: list[RelationshipHit]
    chunks: list[ChunkHit]
    context_text: str
