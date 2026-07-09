from pydantic import BaseModel


class AnswerRequest(BaseModel):
    question: str


class RetrievedContext(BaseModel):
    query: str
    sources: list[str]
    context_text: str


class AnswerResult(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrievals: list[RetrievedContext]
