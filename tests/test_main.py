from types import SimpleNamespace
from unittest.mock import Mock

import main as main_module
from graphtool.retrieval import RetrievalResult, SourceReference


class FakeRuntime:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.graph_store = object()
        self.chunk_store = object()
        self.fast_llm = object()
        self.knowledge_base_store = object()
        self.graph_embedding_store = object()
        self.knowledge_base_embedding_store = object()
        self.chunk_embedding_store = object()
        self.chunk_extraction_store = object()
        self.taxonomy_suggestion_store = object()
        self.search_calls = []

    def search(self, query):
        self.search_calls.append(("direct", query))
        return _result(query, "direct.md", "Direct context.")

    def search_graph(self, query):
        self.search_calls.append(("graph", query))
        return _result(query, "graph.md", "Graph context.")

    def search_hybrid(self, query):
        self.search_calls.append(("hybrid", query))
        return _result(query, "hybrid.md", "Hybrid context.")


def _result(query: str, source: str, context: str) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        sources=[source],
        references=[SourceReference(source=source)],
        chunks=[],
        context_text=context,
    )


def test_format_source_reference_includes_pdf_page_range():
    reference = SourceReference(
        source="documents/manual.pdf",
        page_start=12,
        page_end=14,
    )

    assert main_module._format_source_reference(reference) == (
        "documents/manual.pdf (pp. 12-14)"
    )


def test_main_runs_and_prints_enabled_search_modes(monkeypatch, capsys, tmp_path):
    paths = SimpleNamespace(
        root=tmp_path,
        documents_dir=tmp_path / "documents",
        pdf_conversions_dir=tmp_path / "pdf-conversions",
        dropped_edges_path=tmp_path / "dropped_edges.jsonl",
        logs_dir=tmp_path / "logs",
        visualizations_dir=tmp_path / "visualizations",
    )
    config = SimpleNamespace(entity_resolution_min_candidate_similarity=0.8)
    runtime = FakeRuntime(paths)
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

    main_module.main()

    output = capsys.readouterr().out
    main_module.load_documents.assert_called_once_with(
        paths.documents_dir,
        source_root=paths.root,
        pdf_llm=runtime.fast_llm,
        pdf_cache_dir=paths.pdf_conversions_dir,
    )
    assert runtime.search_calls == [("direct", main_module.QUERY)]
    assert "Sources: direct.md" in output
    assert "Direct context." in output
    assert f"- {visualization_path}" in output
