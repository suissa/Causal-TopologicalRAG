import pytest
from ctrag.benchmarks.statistics import holm_adjust, paired_comparisons, paired_summary

def test_paired_summary_hand_checkable_and_missing_values_are_explicit():
    result = paired_summary([1.0, 2.0, None], [0.0, 1.0, 7.0], seed=1, resamples=100)
    assert result["effect"] == 1.0 and result["n_pairs"] == 2 and result["n_non_applicable"] == 1
    assert result["ci95"] == [1.0, 1.0]

def test_holm_adjustment_is_monotone_and_preserves_none():
    adjusted = holm_adjust({"a": .01, "b": .04, "c": None})
    assert adjusted["a"] == pytest.approx(.02) and adjusted["b"] == pytest.approx(.04) and adjusted["c"] is None

def test_paired_comparison_only_matches_same_query_cells():
    rows = [{"dataset":"d","seed":1,"query_id":"q","mode":"why","k":3,"baseline":"ct","recall_at_k":1},
            {"dataset":"d","seed":1,"query_id":"q","mode":"why","k":3,"baseline":"bm","recall_at_k":0},
            {"dataset":"d","seed":1,"query_id":"unpaired","mode":"why","k":3,"baseline":"ct","recall_at_k":0}]
    assert paired_comparisons(rows, candidate="ct", baselines=["bm"], metrics=["recall_at_k"], resamples=10)[0]["n_pairs"] == 1
