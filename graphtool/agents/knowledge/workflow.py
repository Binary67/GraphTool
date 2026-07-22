from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from graphtool.agents.knowledge.state import AgentResponse
from graphtool.agents.knowledge.workflow_graph import build_workflow_graph
from graphtool.runtime import GraphToolRuntime

DEFAULT_COMPACT_TRIGGER_TOKENS = 32_000
DEFAULT_COMPACT_RECENT_TOKENS = 8_000


class KnowledgeAgent:
    def __init__(
        self,
        model: BaseChatModel,
        runtime: GraphToolRuntime,
        *,
        compact_trigger_tokens: int = DEFAULT_COMPACT_TRIGGER_TOKENS,
        compact_recent_tokens: int = DEFAULT_COMPACT_RECENT_TOKENS,
    ) -> None:
        if compact_trigger_tokens < 1:
            raise ValueError("Compaction trigger token count must be positive.")
        if compact_recent_tokens < 1:
            raise ValueError("Recent conversation token count must be positive.")
        if compact_recent_tokens >= compact_trigger_tokens:
            raise ValueError(
                "Recent conversation token count must be less than the "
                "compaction trigger."
            )
        self._runtime = runtime
        self._checkpointer = InMemorySaver()
        self._graph = build_workflow_graph(
            model,
            runtime,
            self._checkpointer,
            compact_trigger_tokens=compact_trigger_tokens,
            compact_recent_tokens=compact_recent_tokens,
        )

    def ask(self, question: str, *, thread_id: str) -> AgentResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("Thread ID must not be empty.")
        if not self._runtime.knowledge_base_store.exists():
            raise FileNotFoundError(
                "Knowledge base not found. Synchronize documents before asking."
            )

        config = {
            "configurable": {"thread_id": normalized_thread_id},
            "recursion_limit": 50,
        }
        result = self._graph.invoke(
            {
                "messages": [HumanMessage(content=normalized_question)],
                "question": normalized_question,
                "evidence": [],
                "references": [],
                "search_count": 0,
                "research_action": None,
                "proposed_query": None,
                "direct_response": None,
                "evaluation": None,
                "response": None,
            },
            config=config,
        )
        response = result.get("response")
        if not isinstance(response, AgentResponse):
            raise RuntimeError("Knowledge agent completed without a response.")
        self._checkpointer.delete_thread(normalized_thread_id)
        checkpoint_state = {**result, "response": None}
        self._graph.update_state(config, checkpoint_state, as_node="cleanup")
        return response

    def reset(self, thread_id: str) -> None:
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("Thread ID must not be empty.")
        self._checkpointer.delete_thread(normalized_thread_id)


def create_knowledge_agent(
    model: BaseChatModel,
    runtime: GraphToolRuntime,
    *,
    compact_trigger_tokens: int = DEFAULT_COMPACT_TRIGGER_TOKENS,
    compact_recent_tokens: int = DEFAULT_COMPACT_RECENT_TOKENS,
) -> KnowledgeAgent:
    return KnowledgeAgent(
        model,
        runtime,
        compact_trigger_tokens=compact_trigger_tokens,
        compact_recent_tokens=compact_recent_tokens,
    )
