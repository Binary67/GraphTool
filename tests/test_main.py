from types import SimpleNamespace
from unittest.mock import Mock

import main as main_module
from graphtool.agents import AgentResponse
from graphtool.retrieval import SourceReference


class FakeRuntime:
    def __init__(self) -> None:
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


def test_format_source_reference_uses_slide_label_for_powerpoint():
    reference = SourceReference(
        source="documents/slides.PPTX",
        page_start=3,
        page_end=5,
    )

    assert main_module._format_source_reference(reference) == (
        "documents/slides.PPTX (slides 3-5)"
    )

def test_main_runs_terminal_conversation_with_one_memory_thread(
    monkeypatch,
    capsys,
    tmp_path,
):
    paths = SimpleNamespace(
        logs_dir=tmp_path / "logs",
    )
    config = object()
    runtime = FakeRuntime()
    agent_model = object()
    agent = FakeAgent()
    logger = Mock()

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
