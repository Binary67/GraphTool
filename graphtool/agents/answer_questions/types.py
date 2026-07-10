from typing import Literal

from pydantic import BaseModel


class AnswerRequest(BaseModel):
    question: str


class ChunkReference(BaseModel):
    chunk_id: str
    source: str
    index: int
    heading_path: list[str]


class RetrievedContext(BaseModel):
    type: Literal["search"] = "search"
    query: str
    sources: list[str]
    chunk_references: list[ChunkReference]
    context_text: str


class NeighborhoodChunk(ChunkReference):
    text: str


class ChunkNeighborhood(BaseModel):
    type: Literal["chunk_neighborhood"] = "chunk_neighborhood"
    source: str
    chunk_id: str
    previous: NeighborhoodChunk | None
    current: NeighborhoodChunk
    next: NeighborhoodChunk | None


class AnswerResult(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrievals: list[RetrievedContext]
