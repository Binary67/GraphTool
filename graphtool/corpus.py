from collections.abc import Iterable, Mapping
from pathlib import Path

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.markdown import chunk_markdown
from graphtool.graph.generator import combine_knowledge_graphs, generate_knowledge_graph
from graphtool.graph.json_store import JsonGraphStore, JsonKnowledgeBaseStore
from graphtool.graph.types import KnowledgeGraph
from graphtool.llm.base import LLMClient
from graphtool.retrieval.retriever import retrieve_context
from graphtool.retrieval.types import RetrievalResult


def load_markdown_documents(
    directory: str | Path,
    *,
    source_root: str | Path,
) -> dict[str, str]:
    path = Path(directory)
    if not path.exists():
        return {}

    root = Path(source_root)
    documents = {}
    for markdown_path in sorted(path.rglob("*.md")):
        source = markdown_path.relative_to(root).as_posix()
        documents[source] = markdown_path.read_text()
    return documents


def search_knowledge_base(
    query: str,
    graph_store: JsonGraphStore,
    chunk_store: JsonChunkStore,
    *,
    knowledge_base_store: JsonKnowledgeBaseStore | None = None,
    top_nodes: int = 5,
    top_edges: int = 5,
    top_chunks: int = 5,
) -> RetrievalResult:
    graphs = graph_store.load_all()
    chunks = []
    for graph in graphs:
        if graph.metadata is None:
            raise ValueError("Cannot search graph without metadata.source.")
        chunks.extend(chunk_store.load(graph.metadata.source))

    graph = _load_or_rebuild_knowledge_base(graphs, knowledge_base_store)
    return retrieve_context(
        query,
        graph,
        chunks,
        top_nodes=top_nodes,
        top_edges=top_edges,
        top_chunks=top_chunks,
    )


def filter_unprocessed_sources(
    sources: Iterable[str],
    graph_store: JsonGraphStore,
) -> list[str]:
    return [source for source in sources if not graph_store.exists(source)]


def ingest_unprocessed_documents(
    documents: Mapping[str, str],
    graph_store: JsonGraphStore,
    chunk_store: JsonChunkStore,
    llm: LLMClient,
    *,
    max_chars: int = 3000,
    knowledge_base_store: JsonKnowledgeBaseStore | None = None,
) -> list[KnowledgeGraph]:
    graphs = []
    for source, markdown in documents.items():
        if graph_store.exists(source):
            continue

        chunks = chunk_markdown(markdown, source, max_chars=max_chars)
        chunk_store.save(source, chunks)
        graph = generate_knowledge_graph(chunks, source, llm)
        graph_store.save(graph)
        graphs.append(graph)

    if graphs and knowledge_base_store is not None:
        if knowledge_base_store.exists():
            graph = combine_knowledge_graphs([knowledge_base_store.load(), *graphs])
            knowledge_base_store.save(graph)
        else:
            rebuild_knowledge_base(graph_store, knowledge_base_store)

    return graphs


def rebuild_knowledge_base(
    graph_store: JsonGraphStore,
    knowledge_base_store: JsonKnowledgeBaseStore,
) -> KnowledgeGraph:
    graph = combine_knowledge_graphs(graph_store.load_all())
    knowledge_base_store.save(graph)
    return graph


def _load_or_rebuild_knowledge_base(
    graphs: list[KnowledgeGraph],
    knowledge_base_store: JsonKnowledgeBaseStore | None,
) -> KnowledgeGraph:
    if knowledge_base_store is None:
        return combine_knowledge_graphs(graphs)
    if knowledge_base_store.exists():
        return knowledge_base_store.load()

    graph = combine_knowledge_graphs(graphs)
    knowledge_base_store.save(graph)
    return graph
