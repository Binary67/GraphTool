from graphtool.corpus import (
    load_markdown_documents,
    search_knowledge_base,
    synchronize_documents,
)
from graphtool.llm import load_azure_openai_config
from graphtool.run_logging import configure_run_logger
from graphtool.runtime import DEFAULT_MAX_LOG_FILES, create_runtime, default_paths
from graphtool.visualization import export_knowledge_base_visualizations

QUERY = "What can Claude Code do?"


def main() -> None:
    paths = default_paths()
    logger = configure_run_logger(paths.logs_dir, DEFAULT_MAX_LOG_FILES)
    logger.info("Started GraphTool run")

    try:
        config = load_azure_openai_config()
        runtime = create_runtime(config, paths=paths)
        documents = load_markdown_documents(
            runtime.paths.documents_dir,
            source_root=runtime.paths.root,
        )
        logger.info("Loaded %s markdown documents", len(documents))

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

        logger.info("Exporting visualizations")
        visualization_paths = export_knowledge_base_visualizations(
            runtime.graph_store,
            runtime.paths.visualizations_dir,
            knowledge_base_store=runtime.knowledge_base_store,
        )
        logger.info("Exported %s visualizations", len(visualization_paths))

        logger.info("Searching knowledge base")
        result = search_knowledge_base(
            QUERY,
            runtime.graph_store,
            runtime.chunk_store,
            knowledge_base_store=runtime.knowledge_base_store,
            embedding_client=runtime.fast_llm,
            chunk_embedding_store=runtime.chunk_embedding_store,
        )
        logger.info("Search completed with %s sources", len(result.sources))

        print(f"Sources: {', '.join(result.sources) if result.sources else 'None'}")
        print()
        print(result.context_text)
        print()
        print("Visualizations:")
        for path in visualization_paths:
            print(f"- {path}")

        logger.info("Finished GraphTool run")
    except Exception:
        logger.exception("Run failed")
        raise


if __name__ == "__main__":
    main()
