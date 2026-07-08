from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    source: str
    index: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
