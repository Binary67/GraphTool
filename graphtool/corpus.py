from collections.abc import Iterable, Mapping

from graphtool.chunking.json_store import JsonChunkStore
from graphtool.chunking.markdown import chunk_markdown
from graphtool.graph.generator import combine_knowledge_graphs, generate_knowledge_graph
from graphtool.graph.json_store import JsonGraphStore
from graphtool.graph.types import KnowledgeGraph
from graphtool.llm.base import LLMClient
from graphtool.retrieval.retriever import retrieve_context
from graphtool.retrieval.types import RetrievalResult


def search_knowledge_base(
    query: str,
    graph_store: JsonGraphStore,
    chunk_store: JsonChunkStore,
    *,
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

    return retrieve_context(
        query,
        combine_knowledge_graphs(graphs),
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

    return graphs
