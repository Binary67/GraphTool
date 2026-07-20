from graphtool.agents import KnowledgeAgent, create_knowledge_agent
from graphtool.corpus import synchronize_documents
from graphtool.ingestion import load_documents
from graphtool.llm import (
    create_azure_openai_agent_model,
    load_azure_openai_config,
)
from graphtool.retrieval import SourceReference
from graphtool.run_logging import configure_run_logger
from graphtool.runtime import DEFAULT_MAX_LOG_FILES, create_runtime, default_paths
from graphtool.visualization import export_knowledge_base_visualizations

TERMINAL_THREAD_ID = "terminal"
EXIT_COMMANDS = {"exit", "quit"}


def _format_source_reference(reference: SourceReference) -> str:
    if reference.page_start is None:
        return reference.source
    if reference.page_start == reference.page_end:
        return f"{reference.source} (p. {reference.page_start})"
    return f"{reference.source} (pp. {reference.page_start}-{reference.page_end})"


def _run_conversation(agent: KnowledgeAgent) -> None:
    print("GraphTool agent ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.casefold() in EXIT_COMMANDS:
            print("Goodbye.")
            return

        response = agent.ask(question, thread_id=TERMINAL_THREAD_ID)
        label = "Agent (partial)" if response.status == "partial" else "Agent"
        print(f"\n{label}: {response.answer}")
        references = ", ".join(
            _format_source_reference(reference)
            for reference in response.references
        )
        print(f"Sources: {references or 'None'}")


def main() -> None:
    paths = default_paths()
    logger = configure_run_logger(paths.logs_dir, DEFAULT_MAX_LOG_FILES)
    logger.info("Started GraphTool run")

    try:
        config = load_azure_openai_config()
        runtime = create_runtime(config, paths=paths)
        documents = load_documents(
            runtime.paths.documents_dir,
            source_root=runtime.paths.root,
            pdf_llm=runtime.fast_llm,
            pdf_cache_dir=runtime.paths.pdf_conversions_dir,
        )
        logger.info("Loaded %s documents", len(documents))

        sync_result = synchronize_documents(
            documents,
            runtime.graph_store,
            runtime.chunk_store,
            runtime.fast_llm,
            knowledge_base_store=runtime.knowledge_base_store,
            graph_embedding_store=runtime.graph_embedding_store,
            knowledge_base_embedding_store=(
                runtime.knowledge_base_embedding_store
            ),
            chunk_embedding_store=runtime.chunk_embedding_store,
            chunk_extraction_store=runtime.chunk_extraction_store,
            dropped_edges_path=runtime.paths.dropped_edges_path,
            taxonomy_suggestion_store=runtime.taxonomy_suggestion_store,
            min_candidate_similarity=(
                config.entity_resolution_min_candidate_similarity
            ),
        )
        logger.info(
            "Synchronized documents added=%s changed=%s deleted=%s unchanged=%s",
            len(sync_result.added_sources),
            len(sync_result.changed_sources),
            len(sync_result.deleted_sources),
            len(sync_result.unchanged_sources),
        )
        runtime.prepare_search()

        logger.info("Exporting visualizations")
        visualization_paths = export_knowledge_base_visualizations(
            runtime.graph_store,
            runtime.paths.visualizations_dir,
            knowledge_base_store=runtime.knowledge_base_store,
        )
        logger.info("Exported %s visualizations", len(visualization_paths))

        print("Visualizations:")
        for path in visualization_paths:
            print(f"- {path}")

        agent_model = create_azure_openai_agent_model(config)
        agent = create_knowledge_agent(agent_model, runtime)
        _run_conversation(agent)

        logger.info("Finished GraphTool run")
    except Exception:
        logger.exception("Run failed")
        raise


if __name__ == "__main__":
    main()
