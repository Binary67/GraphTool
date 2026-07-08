from pathlib import Path

from graphtool.chunking import JsonChunkStore
from graphtool.corpus import (
    filter_unprocessed_sources,
    ingest_unprocessed_documents,
    search_knowledge_base,
)
from graphtool.graph import JsonGraphStore, combine_knowledge_graphs
from graphtool.llm import AzureOpenAIClient, load_azure_openai_config
from graphtool.run_logging import configure_run_logger
from graphtool.source import source_key
from graphtool.visualization import export_graph_html

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
CHUNKS_DIR = ROOT / "data" / "chunks"
GRAPHS_DIR = ROOT / "data" / "graphs"
LOGS_DIR = ROOT / "logs"
VISUALIZATIONS_DIR = ROOT / "data" / "visualizations"
DOCUMENT_VISUALIZATIONS_DIR = VISUALIZATIONS_DIR / "documents"
KNOWLEDGE_BASE_VISUALIZATION_PATH = VISUALIZATIONS_DIR / "knowledge_graph.html"
MAX_LOG_FILES = 3
QUERY = "What does the knowledge base say about validation?"


def main() -> None:
    logger = configure_run_logger(LOGS_DIR, MAX_LOG_FILES)
    logger.info("Started GraphTool run")

    try:
        graph_store = JsonGraphStore(GRAPHS_DIR)
        chunk_store = JsonChunkStore(CHUNKS_DIR)
        documents = _load_markdown_documents(DOCUMENTS_DIR)
        logger.info("Loaded %s markdown documents", len(documents))

        unprocessed_sources = filter_unprocessed_sources(documents, graph_store)
        logger.info("Found %s unprocessed documents", len(unprocessed_sources))

        if unprocessed_sources:
            logger.info("Ingesting %s documents", len(unprocessed_sources))
            llm = AzureOpenAIClient(load_azure_openai_config())
            ingest_unprocessed_documents(
                {
                    source: documents[source]
                    for source in unprocessed_sources
                },
                graph_store,
                chunk_store,
                llm,
            )
            logger.info("Finished ingesting documents")
        else:
            logger.info("No documents require ingestion")

        logger.info("Exporting visualizations")
        visualization_paths = _export_visualizations(graph_store)
        logger.info("Exported %s visualizations", len(visualization_paths))

        logger.info("Searching knowledge base")
        result = search_knowledge_base(QUERY, graph_store, chunk_store)
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


def _export_visualizations(graph_store: JsonGraphStore) -> list[Path]:
    graphs = graph_store.load_all()
    paths = []

    for graph in graphs:
        if graph.metadata is None:
            raise ValueError("Cannot visualize graph without metadata.source.")

        paths.append(
            export_graph_html(
                graph,
                DOCUMENT_VISUALIZATIONS_DIR
                / f"{source_key(graph.metadata.source)}.html",
            )
        )

    paths.append(
        export_graph_html(
            combine_knowledge_graphs(graphs),
            KNOWLEDGE_BASE_VISUALIZATION_PATH,
        )
    )
    return paths


def _load_markdown_documents(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}

    documents = {}
    for path in sorted(directory.rglob("*.md")):
        source = path.relative_to(ROOT).as_posix()
        documents[source] = path.read_text()
    return documents


if __name__ == "__main__":
    main()
