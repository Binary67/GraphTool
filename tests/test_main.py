from types import SimpleNamespace
from unittest.mock import Mock

import main as main_module
from graphtool.agents import AgentResponse
from graphtool.retrieval import SourceReference


class FakeRuntime:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.graph_store = object()
        self.chunk_store = object()
        self.fast_llm = object()
        self.audio_transcriber = object()
        self.knowledge_base_store = object()
        self.graph_embedding_store = object()
        self.knowledge_base_embedding_store = object()
        self.chunk_embedding_store = object()
        self.chunk_extraction_store = object()
        self.taxonomy_suggestion_store = object()
        self.prepare_search = Mock()


class FakeAgent:
    def __init__(self):
        self.ask_calls = []
        self.responses = [
            AgentResponse(
                answer="First answer.",
                status="complete",
                references=[SourceReference(source="docs/first.md")],
                search_count=1,
            ),
            AgentResponse(
                answer="Follow-up answer.",
                status="partial",
                references=[
                    SourceReference(
                        source="documents/manual.pdf",
                        page_start=2,
                        page_end=3,
                    )
                ],
                search_count=5,
            ),
        ]

    def ask(self, question, *, thread_id):
        self.ask_calls.append((question, thread_id))
        return self.responses.pop(0)


def test_format_source_reference_includes_pdf_page_range():
    reference = SourceReference(
        source="documents/manual.pdf",
        page_start=12,
        page_end=14,
    )

    assert main_module._format_source_reference(reference) == (
        "documents/manual.pdf (pp. 12-14)"
    )


def test_main_runs_terminal_conversation_with_one_memory_thread(
    monkeypatch,
    capsys,
    tmp_path,
):
    paths = SimpleNamespace(
        root=tmp_path,
        documents_dir=tmp_path / "documents",
        pdf_conversions_dir=tmp_path / "pdf-conversions",
        audio_transcriptions_dir=tmp_path / "audio-transcriptions",
        dropped_edges_path=tmp_path / "dropped_edges.jsonl",
        logs_dir=tmp_path / "logs",
        visualizations_dir=tmp_path / "visualizations",
    )
    config = SimpleNamespace(entity_resolution_min_candidate_similarity=0.8)
    runtime = FakeRuntime(paths)
    agent_model = object()
    agent = FakeAgent()
    logger = Mock()
    visualization_path = paths.visualizations_dir / "knowledge-base.html"

    monkeypatch.setattr(main_module, "default_paths", Mock(return_value=paths))
    monkeypatch.setattr(
        main_module,
        "configure_run_logger",
        Mock(return_value=logger),
    )
    monkeypatch.setattr(
        main_module,
        "load_azure_openai_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        main_module,
        "create_runtime",
        Mock(return_value=runtime),
    )
    monkeypatch.setattr(
        main_module,
        "load_documents",
        Mock(return_value={"docs/guide.md": "# Guide\nText."}),
    )
    monkeypatch.setattr(
        main_module,
        "synchronize_documents",
        Mock(
            return_value=SimpleNamespace(
                added_sources=["docs/guide.md"],
                changed_sources=[],
                deleted_sources=[],
                unchanged_sources=[],
            )
        ),
    )
    monkeypatch.setattr(
        main_module,
        "export_knowledge_base_visualizations",
        Mock(return_value=[visualization_path]),
    )
    monkeypatch.setattr(
        main_module,
        "create_azure_openai_agent_model",
        Mock(return_value=agent_model),
    )
    monkeypatch.setattr(
        main_module,
        "create_knowledge_agent",
        Mock(return_value=agent),
    )
    questions = iter(["First question", "Follow-up question", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    main_module.main()

    output = capsys.readouterr().out
    main_module.load_documents.assert_called_once_with(
        paths.documents_dir,
        source_root=paths.root,
        pdf_llm=runtime.fast_llm,
        pdf_cache_dir=paths.pdf_conversions_dir,
        audio_transcriber=runtime.audio_transcriber,
        audio_cache_dir=paths.audio_transcriptions_dir,
    )
    main_module.create_azure_openai_agent_model.assert_called_once_with(config)
    main_module.create_knowledge_agent.assert_called_once_with(agent_model, runtime)
    runtime.prepare_search.assert_called_once_with()
    assert agent.ask_calls == [
        ("First question", main_module.TERMINAL_THREAD_ID),
        ("Follow-up question", main_module.TERMINAL_THREAD_ID),
    ]
    assert "Agent: First answer." in output
    assert "Sources: docs/first.md" in output
    assert "Agent (partial): Follow-up answer." in output
    assert "Sources: documents/manual.pdf (pp. 2-3)" in output
    assert f"- {visualization_path}" in output
