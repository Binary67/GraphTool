import math
from collections.abc import Mapping, Sequence

from graphtool.retrieval.bm25 import BM25Document, BM25Index


def bm25_index(text_by_id: Mapping[str, str]) -> BM25Index:
    return BM25Index(
        [
            BM25Document(id=item_id, text=text)
            for item_id, text in text_by_id.items()
        ]
    )


def bm25_scores(query: str, index: BM25Index) -> dict[str, float]:
    return normalize_scores(
        {document.id: score for document, score in index.rank(query)}
    )


def normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    positive_scores = {
        item_id: score
        for item_id, score in scores.items()
        if score > 0.0
    }
    if not positive_scores:
        return {}
    max_score = max(positive_scores.values())
    return {
        item_id: score / max_score
        for item_id, score in positive_scores.items()
    }


def min_max_normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Stretch scores across [0, 1]. Unlike normalize_scores, this suits
    cosine similarities, whose high floor leaves divide-by-max scores
    compressed near 1.0."""
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {item_id: 1.0 for item_id in scores}
    return {
        item_id: (score - min_score) / (max_score - min_score)
        for item_id, score in scores.items()
    }


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        a * b for a, b in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


def semantic_similarity_scores(
    query_vector: Sequence[float] | None,
    vectors: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    if query_vector is None:
        return {}
    return min_max_normalize_scores(
        {
            item_id: cosine_similarity(query_vector, vector)
            for item_id, vector in vectors.items()
        }
    )
