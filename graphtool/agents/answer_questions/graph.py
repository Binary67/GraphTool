from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from graphtool.agents.answer_questions.prompts import ANSWER_QUESTION_SYSTEM_PROMPT


def build_answer_question_graph(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str = ANSWER_QUESTION_SYSTEM_PROMPT,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        name="answer_question",
    )
