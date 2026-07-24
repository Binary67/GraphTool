import pytest

from graphtool.retrieval.scoring import min_max_normalize_scores


def test_min_max_normalize_scores_empty_input_returns_empty():
    assert min_max_normalize_scores({}) == {}


def test_min_max_normalize_scores_equal_scores_return_one():
    assert min_max_normalize_scores({"single": 0.7}) == {"single": 1.0}
    assert min_max_normalize_scores({"first": 0.7, "second": 0.7}) == {
        "first": 1.0,
        "second": 1.0,
    }


def test_min_max_normalize_scores_stretches_high_floor_scores():
    normalized = min_max_normalize_scores(
        {"refund": 0.82, "shipping": 0.74, "history": 0.65}
    )

    assert normalized["refund"] == pytest.approx(1.0)
    assert normalized["shipping"] == pytest.approx(0.529, abs=0.001)
    assert normalized["history"] == pytest.approx(0.0)


def test_min_max_normalize_scores_handles_negative_minimum():
    normalized = min_max_normalize_scores({"aligned": 0.5, "opposed": -0.5})

    assert normalized == {"aligned": 1.0, "opposed": 0.0}
