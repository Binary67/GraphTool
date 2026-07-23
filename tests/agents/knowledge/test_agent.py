from collections import defaultdict
from itertools import count
import logging

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APITimeoutError

from graphtool.agents.knowledge import create_knowledge_agent
from graphtool.agents.knowledge import workflow_graph
from graphtool.agents.knowledge.prompts import NO_EVIDENCE_ANSWER_SYSTEM_PROMPT
from graphtool.agents.knowledge.state import (
    ConversationSummary,
    FinalAnswerDraft,
    QueryDecomposition,
    SufficiencyDecision,
)
from graphtool.chunking.types import Chunk
from graphtool.retrieval import RetrievalResult, SourceReference
from graphtool.retrieval.types import ChunkHit


class ToolModelResponse:
    pass


class ScriptedRunnable:
    def __init__(self, schema, responses, calls):
        self._schema = schema
        self._responses = responses
        self._calls = calls

    def invoke(self, messages):
        self._calls[self._schema].append(list(messages))
        if not self._responses[self._schema]:
            if self._schema is QueryDecomposition:
                question = str(messages[-1].content).split(
                    "Question to decompose:\n",
                    maxsplit=1,
                )[-1]
                return QueryDecomposition(subquestions=[question])
            raise AssertionError(f"No scripted response for {self._schema.__name__}")
        response = self._responses[self._schema].pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ScriptedModel:
    def __init__(self, responses):
        self.responses = {
            schema: list(values) for schema, values in responses.items()
        }
        self.calls = defaultdict(list)
        self.bound_tools = []
        self.bound_tool_names = []

    def with_structured_output(self, schema):
        self.responses.setdefault(schema, [])
        return ScriptedRunnable(schema, self.responses, self.calls)

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        self.bound_tool_names = [item.name for item in tools]
        self.responses.setdefault(ToolModelResponse, [])
        return ScriptedRunnable(
            ToolModelResponse,
            self.responses,
            self.calls,
        )


class ExistingKnowledgeBaseStore:
    def exists(self):
        return True


class MissingKnowledgeBaseStore:
    def exists(self):
        return False


class FakeRuntime:
    def __init__(self, results, *, neighborhoods=None, knowledge_base_exists=True):
        self.knowledge_base_store = (
            ExistingKnowledgeBaseStore()
            if knowledge_base_exists
            else MissingKnowledgeBaseStore()
        )
        self.results = list(results)
        self.search_calls = []
        self.chunk_store = FakeChunkStore(neighborhoods or {})

    def search(self, query):
        self.search_calls.append(query)
        if not self.results:
            raise AssertionError("No scripted retrieval result")
        return self.results.pop(0)


class FakeChunkStore:
    def __init__(self, neighborhoods):
        self.neighborhoods = neighborhoods
        self.calls = []

    def load_neighborhood(self, source, chunk_id):
        self.calls.append((source, chunk_id))
        return self.neighborhoods[(source, chunk_id)]


def _result(
    query,
    source="docs/guide.md",
    page=1,
    context="Evidence text.",
    chunks=None,
):
    if chunks is None:
        chunk = Chunk(
            id=f"guide-{page:04d}",
            source=source,
            index=page - 1,
            text=context,
            heading_path=["Guide"],
            page_start=page,
            page_end=page,
        )
        chunks = [_chunk_hit(chunk)]
    return RetrievalResult(
        query=query,
        sources=[source],
        references=[
            SourceReference(source=source, page_start=page, page_end=page)
        ],
        chunks=chunks,
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


_tool_call_ids = count(1)


def _tool_call(name, **arguments):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": f"tool-call-{next(_tool_call_ids)}",
                "type": "tool_call",
            }
        ],
    )


def _search_call(query):
    return _tool_call("search_knowledge_base", query=query)


def _neighborhood_call(source, chunk_id):
    return _tool_call(
        "get_chunk_neighborhood",
        source=source,
        chunk_id=chunk_id,
    )


def _direct_response(text):
    return AIMessage(content=text)


def _chunk(chunk_id, index, page, text):
    return Chunk(
        id=chunk_id,
        source="docs/guide.md",
        index=index,
        text=text,
        heading_path=["Guide"],
        page_start=page,
        page_end=page,
    )


def _chunk_hit(chunk):
    return ChunkHit(
        chunk=chunk,
        score=1.0,
        linked_nodes=[],
        linked_relationships=[],
    )


def test_research_tool_selection_is_logged_human_readably(monkeypatch):
    logger = type(
        "FakeLogger",
        (),
        {"info": lambda self, *args: calls.append(args)},
    )()
    calls = []
    monkeypatch.setattr(workflow_graph, "RUN_LOGGER", logger)

    workflow_graph._log_tool_selection(
        {
            "name": "search_knowledge_base",
            "args": {"query": "GraphTool authentication"},
        }
    )
    workflow_graph._log_tool_selection(
        {
            "name": "get_chunk_neighborhood",
            "args": {"source": "guide.pdf", "chunk_id": "guide-0013"},
        }
    )

    assert calls == [
        ("Research selected: %s", "search_knowledge_base"),
        ("Search query: %s", "GraphTool authentication"),
        ("Research selected: %s", "get_chunk_neighborhood"),
        ("Chunk neighborhood: %s :: %s", "guide.pdf", "guide-0013"),
    ]


def test_model_failure_logs_stage_duration_and_status(monkeypatch):
    class FailingModel:
        def invoke(self, messages):
            del messages
            error = RuntimeError("Azure details")
            error.status_code = 429
            raise error

    calls = []
    logger = type(
        "FakeLogger",
        (),
        {
            "info": lambda self, *args: calls.append(("info", *args)),
            "error": lambda self, *args: calls.append(("error", *args)),
        },
    )()
    monkeypatch.setattr(workflow_graph, "RUN_LOGGER", logger)

    with pytest.raises(RuntimeError, match="Azure details"):
        workflow_graph._invoke_model(
            FailingModel(),
            [AIMessage(content="Question")],
            stage="research round 2",
        )

    assert calls[0][0:2] == (
        "info",
        "Starting %s: prompt approximately %d tokens",
    )
    assert calls[0][2] == "research round 2"
    assert calls[1][0:2] == (
        "error",
        "%s failed after %.2fs: %s (status=%s)",
    )
    assert calls[1][2] == "Research round 2"
    assert calls[1][4:] == ("RuntimeError", 429)


def test_agent_retries_answer_with_same_evidence_for_unknown_citation():
    model = ScriptedModel(
        {
            ToolModelResponse: [_search_call("GraphTool capabilities")],
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
            ToolModelResponse: [_search_call("GraphTool capabilities")],
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
            ToolModelResponse: [_search_call("GraphTool capabilities")],
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
            ToolModelResponse: [
                _search_call("Azure OpenAI usage"),
                _search_call("Azure OpenAI decision rationale"),
            ],
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


def test_agent_corrects_missing_follow_up_tool_call():
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call("Azure OpenAI usage"),
                _direct_response("I can answer now."),
                _search_call("Azure OpenAI decision rationale"),
            ],
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
    assert runtime.search_calls == [
        "Azure OpenAI usage",
        "Azure OpenAI decision rationale",
    ]
    corrective_call = model.calls[ToolModelResponse][2]
    assert corrective_call[-1].content == (
        "Your previous response did not call a retrieval tool. The available "
        "evidence is insufficient, so call exactly one retrieval tool now. Do not "
        "answer with prose."
    )


def test_agent_returns_partial_evidence_when_follow_up_research_times_out():
    timeout = APITimeoutError(
        request=httpx.Request("POST", "https://example.openai.azure.com")
    )
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call("Azure OpenAI usage"),
                timeout,
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The reason for the decision is missing.",
                )
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer=(
                        "The available evidence describes its usage, but not the "
                        "decision rationale."
                    ),
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime([_result("Azure OpenAI usage", page=1)])
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("Why do we use Azure OpenAI?", thread_id="thread-1")

    assert response.status == "partial"
    assert response.search_count == 1
    assert response.references[0].page_start == 1
    assert runtime.search_calls == ["Azure OpenAI usage"]


def test_agent_retrieves_allowed_chunk_neighborhood_as_cited_evidence():
    previous = _chunk("guide-0000", 0, 1, "The procedure begins here.")
    current = _chunk("guide-0001", 1, 2, "The matching search passage.")
    next_chunk = _chunk("guide-0002", 2, 3, "The procedure ends here.")
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call("deployment procedure"),
                _neighborhood_call("docs/guide.md", "guide-0001"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The surrounding procedure is missing.",
                ),
                SufficiencyDecision(verdict="sufficient"),
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="The procedure spans all three pages.",
                    cited_reference_ids=["S2", "S1", "S3"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result(
                "deployment procedure",
                page=2,
                chunks=[_chunk_hit(current)],
            )
        ],
        neighborhoods={
            ("docs/guide.md", "guide-0001"): (previous, current, next_chunk)
        },
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask(
        "What is the complete deployment procedure?",
        thread_id="thread-1",
    )

    assert model.bound_tool_names == [
        "search_knowledge_base",
        "get_chunk_neighborhood",
    ]
    schemas = {
        item.name: item.tool_call_schema.model_json_schema()
        for item in model.bound_tools
    }
    assert set(schemas["search_knowledge_base"]["properties"]) == {"query"}
    assert set(schemas["get_chunk_neighborhood"]["properties"]) == {
        "source",
        "chunk_id",
    }
    assert runtime.chunk_store.calls == [("docs/guide.md", "guide-0001")]
    assert response.search_count == 1
    assert response.references == [
        SourceReference(source="docs/guide.md", page_start=1, page_end=1),
        SourceReference(source="docs/guide.md", page_start=2, page_end=2),
        SourceReference(source="docs/guide.md", page_start=3, page_end=3),
    ]


def test_agent_rejects_neighborhood_that_was_not_returned_by_search():
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _neighborhood_call("docs/guide.md", "unknown-chunk"),
                _search_call("GraphTool provider"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="No authorized evidence was retrieved.",
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

    response = agent.ask("Which provider is used?", thread_id="thread-1")

    assert runtime.chunk_store.calls == []
    assert runtime.search_calls == ["GraphTool provider"]
    assert response.search_count == 1


def test_agent_stops_after_two_searches_without_progress(caplog):
    caplog.set_level(logging.INFO, logger=workflow_graph.RUN_LOGGER.name)
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call(f"query {index}") for index in range(1, 3)
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The final decision is not recorded.",
                )
                for _ in range(2)
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
        [_result(f"query {index}", page=index) for index in range(1, 3)]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What was the final decision?", thread_id="thread-1")

    assert response.status == "partial"
    assert response.search_count == 2
    assert runtime.search_calls == ["query 1", "query 2"]
    assert any(
        "Early stopping: unchanged evidence gap after 2 retrievals"
        in record.getMessage()
        for record in caplog.records
    )


def test_agent_researches_each_decomposed_subquestion_and_synthesizes_answer(
    caplog,
):
    caplog.set_level(logging.INFO, logger=workflow_graph.RUN_LOGGER.name)
    model = ScriptedModel(
        {
            QueryDecomposition: [
                QueryDecomposition(
                    subquestions=[
                        "Which provider is used?",
                        "Why was that provider selected?",
                    ]
                )
            ],
            ToolModelResponse: [
                _search_call("GraphTool provider"),
                _search_call("GraphTool provider rationale"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="sufficient"),
                SufficiencyDecision(verdict="sufficient"),
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer=(
                        "GraphTool uses Azure OpenAI because it supports "
                        "structured output."
                    ),
                    cited_reference_ids=["S1", "S2"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result("GraphTool provider", page=1),
            _result("GraphTool provider rationale", page=2),
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask(
        "Which provider is used and why was it selected?",
        thread_id="thread-1",
    )

    assert response.status == "complete"
    assert response.search_count == 2
    assert runtime.search_calls == [
        "GraphTool provider",
        "GraphTool provider rationale",
    ]
    research_calls = model.calls[ToolModelResponse]
    assert "Current subquestion: Which provider is used?" in str(
        research_calls[0][1].content
    )
    assert "Current subquestion: Why was that provider selected?" in str(
        research_calls[1][1].content
    )
    messages = [record.getMessage() for record in caplog.records]
    assert "Decomposed subquestion 1: Which provider is used?" in messages
    assert (
        "Decomposed subquestion 2: Why was that provider selected?"
        in messages
    )


def test_agent_deduplicates_and_scopes_evidence_by_subquestion():
    shared = _chunk("guide-shared", 0, 1, "Shared provider evidence.")
    provider_only = _chunk(
        "guide-provider",
        1,
        2,
        "Provider-only evidence.",
    )
    rationale_only = _chunk(
        "guide-rationale",
        2,
        3,
        "Rationale-only evidence.",
    )
    model = ScriptedModel(
        {
            QueryDecomposition: [
                QueryDecomposition(
                    subquestions=[
                        "Which provider is used?",
                        "Why was it selected?",
                    ]
                )
            ],
            ToolModelResponse: [
                _search_call("provider"),
                _search_call("provider rationale"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="sufficient"),
                SufficiencyDecision(verdict="sufficient"),
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="Azure OpenAI is used for structured output.",
                    cited_reference_ids=["S1", "S2", "S3"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result(
                "provider",
                chunks=[_chunk_hit(shared), _chunk_hit(provider_only)],
            ),
            _result(
                "provider rationale",
                chunks=[_chunk_hit(shared), _chunk_hit(rationale_only)],
            ),
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask(
        "Which provider is used and why?",
        thread_id="thread-1",
    )

    assert response.status == "complete"
    evaluator_prompts = [
        str(call[1].content) for call in model.calls[SufficiencyDecision]
    ]
    assert "Shared provider evidence." in evaluator_prompts[0]
    assert "Provider-only evidence." in evaluator_prompts[0]
    assert "Rationale-only evidence." not in evaluator_prompts[0]
    assert "Shared provider evidence." in evaluator_prompts[1]
    assert "Provider-only evidence." not in evaluator_prompts[1]
    assert "Rationale-only evidence." in evaluator_prompts[1]
    answer_prompt = str(model.calls[FinalAnswerDraft][0][1].content)
    assert answer_prompt.count("Shared provider evidence.") == 1
    second_research_call = model.calls[ToolModelResponse][1]
    assert all(message.type != "tool" for message in second_research_call)
    assert not any(
        "Provider-only evidence." in str(message.content)
        for message in second_research_call
    )


def test_agent_continues_when_missing_information_changes():
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call("decision"),
                _search_call("decision date"),
                _search_call("decision approver"),
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The decision date is missing.",
                ),
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The approver identity is missing.",
                ),
                SufficiencyDecision(verdict="sufficient"),
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="The decision and approver are documented.",
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result("decision", page=1),
            _result("decision date", page=2),
            _result("decision approver", page=3),
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What was decided?", thread_id="thread-1")

    assert response.status == "complete"
    assert response.search_count == 3
    assert runtime.search_calls == [
        "decision",
        "decision date",
        "decision approver",
    ]


def test_agent_stops_at_three_retrievals_when_each_adds_evidence(caplog):
    caplog.set_level(logging.INFO, logger=workflow_graph.RUN_LOGGER.name)

    def result_with_two_chunks(query, start):
        return _result(
            query,
            chunks=[
                _chunk_hit(
                    _chunk(
                        f"guide-{start:04d}",
                        start,
                        start,
                        f"Evidence {start}.",
                    )
                ),
                _chunk_hit(
                    _chunk(
                        f"guide-{start + 1:04d}",
                        start + 1,
                        start + 1,
                        f"Evidence {start + 1}.",
                    )
                ),
            ],
        )

    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call(f"query {index}") for index in range(1, 4)
            ],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The final decision is not recorded.",
                )
                for _ in range(3)
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="The final decision could not be established.",
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            result_with_two_chunks("query 1", 1),
            result_with_two_chunks("query 2", 3),
            result_with_two_chunks("query 3", 5),
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask("What was the final decision?", thread_id="thread-1")

    assert response.status == "partial"
    assert response.search_count == 3
    assert runtime.search_calls == ["query 1", "query 2", "query 3"]
    assert any(
        "Retrieval limit reached after 3 retrievals" in record.getMessage()
        for record in caplog.records
    )


def test_five_subquestions_each_stop_after_two_retrievals_without_progress():
    subquestions = [f"Subquestion {index}?" for index in range(1, 6)]
    queries = [
        f"subquestion {subquestion_index} query {retrieval_index}"
        for subquestion_index in range(1, 6)
        for retrieval_index in range(1, 3)
    ]
    model = ScriptedModel(
        {
            QueryDecomposition: [
                QueryDecomposition(subquestions=subquestions)
            ],
            ToolModelResponse: [_search_call(query) for query in queries],
            SufficiencyDecision: [
                SufficiencyDecision(
                    verdict="insufficient",
                    missing_information="The answer is missing.",
                )
                for _ in queries
            ],
            FinalAnswerDraft: [
                FinalAnswerDraft(
                    answer="The questions could not be fully answered.",
                    cited_reference_ids=["S1"],
                )
            ],
        }
    )
    runtime = FakeRuntime(
        [
            _result(query, page=index)
            for index, query in enumerate(queries, start=1)
        ]
    )
    agent = create_knowledge_agent(model, runtime)

    response = agent.ask(
        "Answer all five parts.",
        thread_id="thread-1",
    )

    assert response.status == "partial"
    assert response.search_count == 10
    assert runtime.search_calls == queries
    answer_prompt = str(model.calls[FinalAnswerDraft][0][1].content)
    assert "Subquestion 1?: insufficient" in answer_prompt
    assert "Subquestion 5?: insufficient" in answer_prompt


def test_agent_discloses_best_effort_answer_after_two_empty_searches():
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call(f"query {index}") for index in range(1, 3)
            ],
            SufficiencyDecision: [
                SufficiencyDecision(verdict="sufficient") for _ in range(2)
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
        [_empty_result(f"query {index}") for index in range(1, 3)]
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
    assert response.search_count == 2
    assert runtime.search_calls == ["query 1", "query 2"]
    answer_call = model.calls[FinalAnswerDraft][0]
    assert answer_call[0].content == NO_EVIDENCE_ANSWER_SYSTEM_PROMPT


def test_evaluator_prevents_substantive_response_without_evidence():
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _direct_response("It uses Azure."),
                _search_call("GraphTool provider"),
            ],
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
            ToolModelResponse: [
                _direct_response("Hello! How can I help?")
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


def test_in_memory_threads_retain_only_their_own_conversation(caplog):
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _direct_response("First answer"),
                _direct_response("Follow-up answer"),
                _direct_response("Separate answer"),
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
    research_calls = model.calls[ToolModelResponse]
    follow_up_text = "\n".join(str(message.content) for message in research_calls[1])
    separate_text = "\n".join(str(message.content) for message in research_calls[2])
    assert "First answer" in follow_up_text
    assert "First answer" not in separate_text
    assert not any(
        "Deserializing unregistered type" in record.getMessage()
        for record in caplog.records
    )


def test_search_budget_resets_for_each_turn_in_the_same_thread():
    model = ScriptedModel(
        {
            ToolModelResponse: [
                _search_call("first query"),
                _search_call("follow-up query"),
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
            ToolModelResponse: [
                _direct_response("First answer"),
                _direct_response("Second answer"),
                _direct_response("Third answer"),
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
        compaction_trigger_tokens=40,
        retained_recent_tokens=20,
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

    research_calls = model.calls[ToolModelResponse]
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
            ToolModelResponse: [_search_call("GraphTool capabilities")],
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
    assert state["response"] is None
    assert [message.content for message in state["messages"]] == [
        "What does GraphTool do?",
        "GraphTool builds a knowledge graph.",
    ]


def test_reset_deletes_conversation_checkpoint():
    model = ScriptedModel(
        {
            ToolModelResponse: [_direct_response("First answer")],
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
