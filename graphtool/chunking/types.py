from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    id: str
    source: str
    index: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None

    @model_validator(mode="after")
    def validate_page_range(self) -> "Chunk":
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError(
                "page_start and page_end must both be set or both be omitted"
            )
        if self.page_start is not None:
            assert self.page_end is not None
            if self.page_start < 1:
                raise ValueError("page_start must be positive")
            if self.page_end < self.page_start:
                raise ValueError(
                    "page_end must be greater than or equal to page_start"
                )
        return self
