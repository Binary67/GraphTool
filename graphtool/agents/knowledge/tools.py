from pydantic import BaseModel, ConfigDict, Field

from graphtool.retrieval import SourceReference
from graphtool.runtime import GraphToolRuntime


class KnowledgeSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    context_text: str
    references: list[SourceReference] = Field(default_factory=list)


def search_knowledge_base(
    runtime: GraphToolRuntime,
    query: str,
) -> KnowledgeSearchResult:
    """Search chunks and knowledge-graph paths relevant to one focused query."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Knowledge base search query must not be empty.")

    result = runtime.search_hybrid(normalized_query)
    return KnowledgeSearchResult(
        query=normalized_query,
        context_text=result.context_text,
        references=result.references,
    )
