from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from graphtool.retrieval import SourceReference


class ResearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search", "respond"]
    query: str | None = None
    response: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ResearchDecision":
        if self.action == "search":
            if not _has_text(self.query):
                raise ValueError("query is required when action is search")
            if self.response is not None:
                raise ValueError("response must be omitted when action is search")
        if self.action == "respond":
            if not _has_text(self.response):
                raise ValueError("response is required when action is respond")
            if self.query is not None:
                raise ValueError("query must be omitted when action is respond")
        return self


class ResearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, pattern=r"\S")


class SufficiencyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["conversation", "sufficient", "insufficient"]
    missing_information: str = ""


class FinalAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, pattern=r"\S")
    cited_reference_ids: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000, pattern=r"\S")


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    reference: SourceReference


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    context_text: str
    reference_ids: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    status: Literal["complete", "partial"]
    references: list[SourceReference] = Field(default_factory=list)
    search_count: int


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: str
    question: str
    evidence: list[EvidenceRecord]
    references: list[EvidenceReference]
    search_count: int
    research_action: Literal["search", "respond"] | None
    proposed_query: str | None
    direct_response: str | None
    evaluation: SufficiencyDecision | None
    response: AgentResponse | None


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())
