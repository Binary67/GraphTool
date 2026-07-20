import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class BM25Document:
    id: str
    text: str


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(
        self,
        documents: Sequence[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._documents = list(documents)
        self._k1 = k1
        self._b = b
        self._tokens_by_id = {
            document.id: tokenize(document.text) for document in self._documents
        }
        self._term_counts_by_id = {
            document_id: Counter(tokens)
            for document_id, tokens in self._tokens_by_id.items()
        }
        lengths = [len(tokens) for tokens in self._tokens_by_id.values()]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0
        self._idf = self._build_idf()

    def rank(self, query: str) -> list[tuple[BM25Document, float]]:
        query_tokens = tokenize(query)
        scored = [
            (index, document, self._score(query_tokens, document.id))
            for index, document in enumerate(self._documents)
        ]
        scored.sort(key=lambda item: (-item[2], item[0]))
        return [(document, score) for _, document, score in scored]

    def score(self, query: str, document_id: str) -> float:
        return self._score(tokenize(query), document_id)

    def _score(self, query_tokens: list[str], document_id: str) -> float:
        if not query_tokens or self._average_length == 0:
            return 0.0

        term_counts = self._term_counts_by_id.get(document_id)
        if term_counts is None:
            return 0.0

        document_length = len(self._tokens_by_id[document_id])
        score = 0.0
        for term in query_tokens:
            frequency = term_counts[term]
            if frequency == 0:
                continue
            numerator = frequency * (self._k1 + 1)
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * document_length / self._average_length
            )
            score += self._idf.get(term, 0.0) * numerator / denominator

        return score

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self._documents)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens_by_id.values():
            document_frequency.update(set(tokens))

        return {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }
