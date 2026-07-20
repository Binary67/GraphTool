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
