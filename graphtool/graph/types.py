from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Node(BaseModel):
    id: str
    label: str
    type: str
    suggested_type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)
    provenance: list["NodeProvenance"] = Field(default_factory=list)


class Edge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)
    provenance: list["EdgeProvenance"] = Field(default_factory=list)


class NodeProvenance(BaseModel):
    source: str
    content_hash: str
    node_id: str
    label: str
    type: str
    suggested_type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)
    resolution_aliases: list[str] = Field(default_factory=list)


class EdgeProvenance(BaseModel):
    source: str
    content_hash: str
    edge_id: str
    source_node_id: str
    target_node_id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)


class GraphMetadata(BaseModel):
    source: str
    content_hash: str
    model: str | None = None
    created_at: datetime


class KnowledgeGraph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
    metadata: GraphMetadata | None = None

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "KnowledgeGraph":
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("node ids must be unique")

        edge_ids = {edge.id for edge in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("edge ids must be unique")

        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(
                    f"edge {edge.id!r} references missing source node {edge.source!r}"
                )
            if edge.target not in node_ids:
                raise ValueError(
                    f"edge {edge.id!r} references missing target node {edge.target!r}"
                )

        return self
