from graphtool.retrieval.bm25 import BM25Document, BM25Index


def test_rank_scores_exact_match_above_unrelated_document():
    index = BM25Index(
        [
            BM25Document(id="python", text="Python validation library"),
            BM25Document(id="coffee", text="Coffee brewing guide"),
        ]
    )

    ranked = index.rank("python validation")

    assert ranked[0][0].id == "python"
    assert ranked[0][1] > ranked[1][1]


def test_rank_empty_query_returns_zero_scores_in_stable_order():
    index = BM25Index(
        [
            BM25Document(id="first", text="Python"),
            BM25Document(id="second", text="Validation"),
        ]
    )

    ranked = index.rank("")

    assert [document.id for document, _ in ranked] == ["first", "second"]
    assert [score for _, score in ranked] == [0.0, 0.0]


def test_rank_preserves_document_order_for_equal_scores():
    index = BM25Index(
        [
            BM25Document(id="first", text="Python library"),
            BM25Document(id="second", text="Python library"),
        ]
    )

    ranked = index.rank("python")

    assert [document.id for document, _ in ranked] == ["first", "second"]
