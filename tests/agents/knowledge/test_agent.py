from collections import defaultdict

import pytest

from graphtool.agents.knowledge import create_knowledge_agent
from graphtool.agents.knowledge.prompts import NO_EVIDENCE_ANSWER_SYSTEM_PROMPT
from graphtool.agents.knowledge.state import (
    ConversationSummary,
    FinalAnswerDraft,
    ResearchDecision,
    ResearchQuery,
    SufficiencyDecision,
)
from graphtool.retrieval import RetrievalResult, SourceReference


class ScriptedRunnable:
    def __init__(self, schema, responses, calls):
        self._schema = schema
        self._responses = responses
        self._calls = calls

    def invoke(self, messages):
        self._calls[self._schema].append(list(messages))
        if not self._responses[self._schema]:
            raise AssertionError(f"No scripted response for {self._schema.__name__}")
        return self._responses[self._schema].pop(0)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = {
            schema: list(values) for schema, values in responses.items()
        }
        self.calls = defaultdict(list)

    def with_structured_output(self, schema):
        self.responses.setdefault(schema, [])
        return ScriptedRunnable(schema, self.responses, self.calls)


class ExistingKnowledgeBaseStore:
    def exists(self):
        return True


class MissingKnowledgeBaseStore:
    def exists(self):
        return False


class FakeRuntime:
    def __init__(self, results, *, knowledge_base_exists=True):
        self.knowledge_base_store = (
            ExistingKnowledgeBaseStore()
            if knowledge_base_exists
            else MissingKnowledgeBaseStore()
        )
        self.results = list(results)
        self.search_calls = []

    def search(self, query):
        self.search_calls.append(query)
        if not self.results:
            raise AssertionError("No scripted retrieval result")
        return self.results.pop(0)


def _result(query, source="docs/guide.md", page=1, context="Evidence text."):
    return RetrievalResult(
        query=query,
        sources=[source],
        references=[
            SourceReference(source=source, page_start=page, page_end=page)
        ],
        chunks=[],
        context_text=context,
    )


def _empty_result(query):
    return RetrievalResult(
        query=query,
        sources=[],
        references=[],
        chunks=[],
        context_text=f"Query: {query}\n\nEvidence:\n- None",
    )


def test_research_decision_rejects_response_for_search_action():
    with pytest.raises(
        ValueError,
        match="response must be omitted when action is search",
    ):
        ResearchDecision(
            action="search",
            query="GraphTool capabilities",
            response="GraphTool builds a knowledge graph.",
        )


def test_research_decision_rejects_query_for_respond_action():
    with pytest.raises(
        ValueError,
        match="query must be omitted when action is respond",
    ):
        ResearchDecision(
            action="respond",
            query="GraphTool capabilities",
            response="Hello!",
        )


def test_agent_retries_answer_with_same_evidence_for_unknown_citation():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="search", query="GraphTool capabilities")
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="sufficient")
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="GraphTool builds a knowledge graph for S999.",
                    cited_reference_ids=["S1", "S999"],
                ),
                FinalAnswerDraft(
                    answer="GraphTool builds a knowledge graph.",
                    cited_reference_ids=["S1"],
                ),
            ],
        }
    )
    runtime = FakeRuntime([_result("GraphTool capabilities")])
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What does GraphTool do?", thread_id="thread-1")

    assert response.answer == "GraphTool builds a knowledge graph."
    assert response.status == "complete"
    assert response.search_count == 1
    assert response.references == [
        SourceReference(
            source="docs/guide.md",
            page_start=1,
            page_end=1,
        )
    ]
    assert runtime.search_calls == ["GraphTool capabilities"]
    answer_calls = model.calls[FinalAnswerDraft]
    assert len(answer_calls) == 2
    assert answer_calls[0][1].content == answer_calls[1][1].content
    assert "unknown reference IDs: S999" in answer_calls[1][0].content
    assert "available reference IDs: S1" in answer_calls[1][0].content


def test_agent_does_not_retry_answer_when_all_citations_are_valid():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="search", query="GraphTool capabilities")
            ],
            SufficiencyDecision: [SufficiencyDecision(verdict="sufficient")],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="GraphTool builds a knowledge graph.",
                    cited_reference_ids=["S1", "S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime([_result("GraphTool capabilities")])
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What does GraphTool do?", thread_id="thread-1")

    assert response.references == [
        SourceReference(
            source="docs/guide.md",
            page_start=1,
            page_end=1,
        )
    ]
    assert response.search_count == 1
    assert runtime.search_calls == ["GraphTool capabilities"]
    assert len(model.calls[FinalAnswerDraft]) == 1


def test_agent_fails_after_retry_repeats_unknown_citation():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="search", query="GraphTool capabilities")
            ],
            SufficiencyDecision: [SufficiencyDecision(verdict="sufficient")],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="First unsupported answer.",
                    cited_reference_ids=["S999"],
                ),
                FinalAnswerDraft(
                    answer="Second unsupported answer.",
                    cited_reference_ids=["S998"],
                ),
            ],
        }
    )
    runtime = FakeRuntime([_result("GraphTool capabilities")])
    agent = create_knowledge_agent(model, runtime)

    with pytest.raises(
        RuntimeError,
        match="unknown references after retry: S998",
    ):
        agent.ask("What does GraphTool do?", thread_id="thread-1")

    assert runtime.search_calls == ["GraphTool capabilities"]
    assert len(model.calls[FinalAnswerDraft]) == 2


def test_agent_reformulates_search_after_insufficient_evidence():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="search", query="Azure OpenAI usage")
            ],
            ResearchQuery: [ResearchQuery(query="Azure OpenAI decision rationale")],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The reason for the decision is missing.",
                ),
                SufficiencyDecision(verdict="sufficient"),
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="It was selected for structured output support.",
                    cited_reference_ids=["S2"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result("Azure OpenAI usage", page=1),
            _result("Azure OpenAI decision rationale", page=2),
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("Why do we use Azure OpenAI?", thread_id="thread-1")

    assert response.status == "complete"
    assert response.search_count == 2
    assert runtime.search_calls == [
        "Azure OpenAI usage",
        "Azure OpenAI decision rationale",
    ]
    assert response.references[0].page_start == 2


def test_agent_stops_after_five_searches_and_returns_partial_answer():
    model = ScriptedModel(
        {
            ResearchDecision: [ResearchDecision(action="search", query="query 1")],
            ResearchQuery: [
                ResearchQuery(query=f"query {index}") for index in range(2, 6)
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The final decision is not recorded.",
                )
                for _ in range(5)
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer=(
                        "The options are documented, but the final decision could "
                        "not be established."
                    ),
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [_result(f"query {index}", page=index) for index in range(1, 6)]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What was the final decision?", thread_id="thread-1")

    assert response.status == "partial"
    assert response.search_count == 5
    assert runtime.search_calls == [f"query {index}" for index in range(1, 6)]


def test_agent_discloses_best_effort_answer_after_five_empty_searches():
    model = ScriptedModel(
        {
            ResearchDecision: [ResearchDecision(action="search", query="query 1")],
            ResearchQuery: [
                ResearchQuery(query=f"query {index}") for index in range(2, 6)
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="sufficient") for _ in range(5)
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="A best-effort general-knowledge answer.",
                    cited_reference_ids=[],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [_empty_result(f"query {index}") for index in range(1, 6)]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What happened?", thread_id="thread-1")

    assert response.answer == (
        "I couldn't find supporting information in the knowledge base. The "
        "following is a best-effort answer based on general knowledge and is not "
        "verified against the knowledge base.\n\n"
        "A best-effort general-knowledge answer."
    )
    assert response.status == "partial"
    assert response.references == []
    assert response.search_count == 5
    assert runtime.search_calls == [f"query {index}" for index in range(1, 6)]
    answer_call = model.calls[FinalAnswerDraft][0]
    assert answer_call[0].content == NO_EVIDENCE_ANSWER_SYSTEM_PROMPT


def test_evaluator_prevents_substantive_response_without_evidence():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="respond", response="It uses Azure.")
            ],
            ResearchQuery: [ResearchQuery(query="GraphTool provider")],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="No evidence was retrieved.",
                ),
                SufficiencyDecision(verdict="sufficient"),
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="GraphTool uses Azure OpenAI.",
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime([_result("GraphTool provider")])
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("Which provider does GraphTool use?", thread_id="thread-1")

    assert response.answer == "GraphTool uses Azure OpenAI."
    assert response.search_count == 1
    assert runtime.search_calls == ["GraphTool provider"]


def test_agent_allows_evaluator_approved_conversation_without_search():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="respond", response="Hello! How can I help?")
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="conversation")
            ],
        }
    )
    runtime = FakeRuntime([])
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("Hello", thread_id="thread-1")

    assert response.answer == "Hello! How can I help?"
    assert response.status == "complete"
    assert response.references == []
    assert response.search_count == 0
    assert runtime.search_calls == []


def test_in_memory_threads_retain_only_their_own_conversation():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="respond", response="First answer"),
                ResearchDecision(action="respond", response="Follow-up answer"),
                ResearchDecision(action="respond", response="Separate answer"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="conversation") for _ in range(3)
            ],
        }
    )
    runtime = FakeRuntime([])
    agent = create_knowledge_agent(model, runtime)

    first = agent.ask("Hello", thread_id="thread-a")
    follow_up = agent.ask("Thanks", thread_id="thread-a")
    separate = agent.ask("Hello", thread_id="thread-b")

    assert first.search_count == 0
    assert follow_up.search_count == 0
    assert separate.search_count == 0
    research_calls = model.calls[ResearchDecision]
    follow_up_text = "\n".join(str(message.content) for message in research_calls[1])
    separate_text = "\n".join(str(message.content) for message in research_calls[2])
    assert "First answer" in follow_up_text
    assert "First answer" not in separate_text


def test_search_budget_resets_for_each_turn_in_the_same_thread():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="search", query="first query"),
                ResearchDecision(action="search", query="follow-up query"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="sufficient") for _ in range(2)
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(answer="First answer", cited_reference_ids=["S1"]),
                FinalAnswerDraft(
                    answer="Follow-up answer",
                    cited_reference_ids=["S1"],
                ),
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result("first query", page=1),
            _result("follow-up query", page=2),
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    first = agent.ask("First question", thread_id="thread-a")
    follow_up = agent.ask("Follow-up question", thread_id="thread-a")

    assert first.search_count == 1
    assert follow_up.search_count == 1
    assert follow_up.references[0].page_start == 2


def test_agent_incrementally_compacts_old_conversation_messages():
    model = ScriptedModel(
        {
            ConversationSummary: [
                ConversationSummary(summary="Apollo summary version one."),
                ConversationSummary(summary="Apollo summary version two."),
            ],
            ResearchDecision: [
                ResearchDecision(action="respond", response="First answer"),
                ResearchDecision(action="respond", response="Second answer"),
                ResearchDecision(action="respond", response="Third answer"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="conversation") for _ in range(3)
            ],
        }
    )
    runtime = FakeRuntime([])
    agent = create_knowledge_agent(
        model,
        runtime,
        compact_trigger_tokens=40,
        compact_recent_tokens=20,
    )
    first_question = "Apollo initial context " * 20
    second_question = "Apollo follow-up context " * 20

    agent.ask(first_question, thread_id="thread-a")
    agent.ask(second_question, thread_id="thread-a")
    agent.ask("Thanks", thread_id="thread-a")

    summary_calls = model.calls[ConversationSummary]
    assert len(summary_calls) == 2
    first_summary_input = str(summary_calls[0][1].content)
    second_summary_input = str(summary_calls[1][1].content)
    assert first_question.strip() in first_summary_input
    assert "First answer" in first_summary_input
    assert "Apollo summary version one." in second_summary_input
    assert second_question.strip() in second_summary_input
    assert "Second answer" in second_summary_input

    research_calls = model.calls[ResearchDecision]
    second_research_text = "\n".join(
        str(message.content) for message in research_calls[1]
    )
    third_research_text = "\n".join(
        str(message.content) for message in research_calls[2]
    )
    assert "Apollo summary version one." in second_research_text
    assert first_question.strip() not in second_research_text
    assert "Apollo summary version two." in third_research_text
    assert second_question.strip() not in third_research_text


def test_completed_turn_keeps_one_clean_checkpoint():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="search", query="GraphTool capabilities")
            ],
            SufficiencyDecision: [SufficiencyDecision(verdict="sufficient")],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="GraphTool builds a knowledge graph.",
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime([_result("GraphTool capabilities")])
    agent = create_knowledge_agent(model, runtime)
    config = {"configurable": {"thread_id": "thread-a"}}

    response = agent.ask("What does GraphTool do?", thread_id="thread-a")

    checkpoints = list(agent._checkpointer.list(config))
    state = agent._graph.get_state(config).values
    assert len(checkpoints) == 1
    assert state["question"] == ""
    assert state["evidence"] == []
    assert state["references"] == []
    assert state["search_count"] == 0
    assert state["research_action"] is None
    assert state["evaluation"] is None
    assert state["response"] == response
    assert [message.content for message in state["messages"]] == [
        "What does GraphTool do?",
        "GraphTool builds a knowledge graph.",
    ]


def test_reset_deletes_conversation_checkpoint():
    model = ScriptedModel(
        {
            ResearchDecision: [
                ResearchDecision(action="respond", response="First answer")
            ],
            SufficiencyDecision: [SufficiencyDecision(verdict="conversation")],
        }
    )
    runtime = FakeRuntime([])
    agent = create_knowledge_agent(model, runtime)
    config = {"configurable": {"thread_id": "thread-a"}}
    agent.ask("Hello", thread_id="thread-a")

    agent.reset("thread-a")

    assert list(agent._checkpointer.list(config)) == []


def test_reset_rejects_empty_thread_id():
    agent = create_knowledge_agent(ScriptedModel({}), FakeRuntime([]))

    with pytest.raises(ValueError, match="Thread ID must not be empty"):
        agent.reset(" ")


def test_agent_rejects_invalid_input_and_missing_knowledge_base():
    model = ScriptedModel({})
    runtime = FakeRuntime([], knowledge_base_exists=False)
    agent = create_knowledge_agent(model, runtime)

    try:
        agent.ask("", thread_id="thread-1")
    except ValueError as exc:
        assert str(exc) == "Question must not be empty."
    else:
        raise AssertionError("Expected empty question to fail")

    try:
        agent.ask("Question", thread_id=" ")
    except ValueError as exc:
        assert str(exc) == "Thread ID must not be empty."
    else:
        raise AssertionError("Expected empty thread ID to fail")

    try:
        agent.ask("Question", thread_id="thread-1")
    except FileNotFoundError as exc:
        assert "Knowledge base not found" in str(exc)
    else:
        raise AssertionError("Expected missing knowledge base to fail")
