from pathlib import Path

from graphtool.chunking import JsonChunkStore
from graphtool.corpus import (
    filter_unprocessed_sources,
    ingest_unprocessed_documents,
    search_knowledge_base,
)
from graphtool.graph import JsonGraphStore
from graphtool.llm import AzureOpenAIClient, load_azure_openai_config

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
CHUNKS_DIR = ROOT / "data" / "chunks"
GRAPHS_DIR = ROOT / "data" / "graphs"
QUERY = "What does the knowledge base say about validation?"


def main() -> None:
    graph_store = JsonGraphStore(GRAPHS_DIR)
    chunk_store = JsonChunkStore(CHUNKS_DIR)
    documents = _load_markdown_documents(DOCUMENTS_DIR)
    unprocessed_sources = filter_unprocessed_sources(documents, graph_store)

    if unprocessed_sources:
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

    result = search_knowledge_base(QUERY, graph_store, chunk_store)
    print(f"Sources: {', '.join(result.sources) if result.sources else 'None'}")
    print()
    print(result.context_text)


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
