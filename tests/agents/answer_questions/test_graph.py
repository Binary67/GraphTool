from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from graphtool.agents.answer_questions.graph import build_answer_question_graph
from graphtool.agents.answer_questions.runner import MAX_AGENT_ITERATIONS


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_runs_search_then_neighborhood_within_recursion_limit():
    calls = []

    def search(query: str) -> str:
        calls.append(("search", query))
        return '{"type":"search"}'

    def neighborhood(source: str, chunk_id: str) -> str:
        calls.append(("neighborhood", source, chunk_id))
        return '{"type":"chunk_neighborhood"}'

    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_knowledge_context",
                        "args": {"query": "target topic"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_neighborhood",
                        "args": {
                            "source": "docs/guide.md",
                            "chunk_id": "guide-chunk-0001",
                        },
                        "id": "neighborhood-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Final answer"),
        ]
    )
    tools = [
        StructuredTool.from_function(
            search,
            name="retrieve_knowledge_context",
            description="Search for evidence.",
        ),
        StructuredTool.from_function(
            neighborhood,
            name="get_chunk_neighborhood",
            description="Load adjacent chunks.",
        ),
    ]
    graph = build_answer_question_graph(model, tools)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Question"}]},
        config={"recursion_limit": MAX_AGENT_ITERATIONS},
    )

    assert calls == [
        ("search", "target topic"),
        ("neighborhood", "docs/guide.md", "guide-chunk-0001"),
    ]
    assert result["messages"][-1].content == "Final answer"
