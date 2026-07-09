from pathlib import Path

from graphtool.chunking import JsonChunkStore
from graphtool.corpus import (
    filter_unprocessed_sources,
    ingest_unprocessed_documents,
    load_markdown_documents,
    search_knowledge_base,
)
from graphtool.graph import (
    JsonEmbeddingStore,
    JsonGraphEmbeddingStore,
    JsonGraphStore,
    JsonKnowledgeBaseStore,
    JsonTaxonomySuggestionStore,
)
from graphtool.llm import AzureOpenAIClient, load_azure_openai_config
from graphtool.run_logging import configure_run_logger
from graphtool.visualization import export_knowledge_base_visualizations

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
CHUNKS_DIR = ROOT / "data" / "chunks"
GRAPHS_DIR = ROOT / "data" / "graphs"
GRAPH_EMBEDDINGS_DIR = ROOT / "data" / "graph_embeddings"
KNOWLEDGE_BASE_PATH = ROOT / "data" / "knowledge_base.json"
KNOWLEDGE_BASE_EMBEDDINGS_PATH = ROOT / "data" / "knowledge_base_embeddings.json"
TAXONOMY_SUGGESTIONS_PATH = ROOT / "data" / "taxonomy_suggestions.json"
DROPPED_EDGES_PATH = ROOT / "data" / "dropped_edges.jsonl"
LOGS_DIR = ROOT / "logs"
VISUALIZATIONS_DIR = ROOT / "data" / "visualizations"
MAX_LOG_FILES = 3
QUERY = "What can Claude Code do?"


def main() -> None:
    logger = configure_run_logger(LOGS_DIR, MAX_LOG_FILES)
    logger.info("Started GraphTool run")

    try:
        graph_store = JsonGraphStore(GRAPHS_DIR)
        knowledge_base_store = JsonKnowledgeBaseStore(KNOWLEDGE_BASE_PATH)
        graph_embedding_store = JsonGraphEmbeddingStore(GRAPH_EMBEDDINGS_DIR)
        knowledge_base_embedding_store = JsonEmbeddingStore(
            KNOWLEDGE_BASE_EMBEDDINGS_PATH
        )
        taxonomy_suggestion_store = JsonTaxonomySuggestionStore(
            TAXONOMY_SUGGESTIONS_PATH
        )
        chunk_store = JsonChunkStore(CHUNKS_DIR)
        documents = load_markdown_documents(DOCUMENTS_DIR, source_root=ROOT)
        logger.info("Loaded %s markdown documents", len(documents))

        unprocessed_sources = filter_unprocessed_sources(documents, graph_store)
        logger.info("Found %s unprocessed documents", len(unprocessed_sources))

        if unprocessed_sources:
            logger.info("Ingesting %s documents", len(unprocessed_sources))
            config = load_azure_openai_config()
            llm = AzureOpenAIClient(
                config,
                text_deployment=config.fast_deployment,
            )
            ingest_unprocessed_documents(
                {
                    source: documents[source]
                    for source in unprocessed_sources
                },
                graph_store,
                chunk_store,
                llm,
                knowledge_base_store=knowledge_base_store,
                graph_embedding_store=graph_embedding_store,
                knowledge_base_embedding_store=knowledge_base_embedding_store,
                dropped_edges_path=DROPPED_EDGES_PATH,
                taxonomy_suggestion_store=taxonomy_suggestion_store,
                min_candidate_similarity=(
                    config.entity_resolution_min_candidate_similarity
                ),
            )
            logger.info("Finished ingesting documents")
        else:
            logger.info("No documents require ingestion")

        logger.info("Exporting visualizations")
        visualization_paths = export_knowledge_base_visualizations(
            graph_store,
            VISUALIZATIONS_DIR,
            knowledge_base_store=knowledge_base_store,
        )
        logger.info("Exported %s visualizations", len(visualization_paths))

        logger.info("Searching knowledge base")
        result = search_knowledge_base(
            QUERY,
            graph_store,
            chunk_store,
            knowledge_base_store=knowledge_base_store,
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
