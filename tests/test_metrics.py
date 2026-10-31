import math

from app.metrics import (average_precision, f1_at_k, ndcg_at_k, precision_at_k,
                         recall_at_k, reciprocal_rank)


def test_precision_and_recall_at_k():
    ranked, relevant = [1, 2, 3, 4], {1, 3}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert math.isclose(f1_at_k(ranked, relevant, 2), 0.5)


def test_average_precision_known_value():
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert math.isclose(average_precision([1, 2, 3, 4], {1, 3}), (1 + 2 / 3) / 2)


def test_average_precision_penalises_unretrieved_relevant_documents():
    assert math.isclose(average_precision([1, 2], {1, 99}), 0.5)


def test_reciprocal_rank():
    assert reciprocal_rank([5, 1, 2], {1}) == 0.5
    assert reciprocal_rank([5, 6], {1}) == 0.0


def test_ndcg_is_one_for_ideal_ranking_and_lower_otherwise():
    assert math.isclose(ndcg_at_k([1, 2, 9], {1, 2}, 3), 1.0)
    assert ndcg_at_k([9, 1, 2], {1, 2}, 3) < 1.0
    assert ndcg_at_k([1, 2], set(), 3) == 0.0


def test_empty_relevant_set_gives_zero():
    assert average_precision([1, 2], set()) == 0.0
    assert recall_at_k([1, 2], set(), 2) == 0.0
