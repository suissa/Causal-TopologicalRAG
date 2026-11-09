from ctrag.benchmarks.codemanager_paired import (
    candidate_universe, legacy_proxy_rank, run_paired, _metrics,
)
from ctrag.benchmarks.codemanager_perturbations import fixture, signal
from ctrag.experiments.codemanager import rank_hypotheses


def test_legacy_proxy_is_deterministic():
    signals = fixture()
    assert legacy_proxy_rank(signals) == legacy_proxy_rank(reversed(signals))


def test_paired_report_has_same_scenario_denominator_and_candidate_universe():
    report = run_paired()
    assert report["scenario_count"] == report["denominator"] == 12
    assert set(report["aggregate"]) == {"legacy_proxy", "evidence_aware"}
    assert set(report["delta_evidence_aware_minus_legacy"]) == {
        "gold_in_candidates", "top1", "recall_at_1", "recall_at_2", "mrr"
    }
    assert report["baseline_is_historical_implementation"] is False
    for row in report["scenarios"].values():
        universe = set(row["candidate_universe"])
        assert set(row["legacy_order"]) == universe
        assert set(row["evidence_aware_order"]) == universe
        assert row["candidate_count"] == len(universe)


def test_metrics_fail_closed_if_ranker_drops_or_invents_candidate():
    universe = candidate_universe(fixture())
    try:
        _metrics(["deploy-a"], universe, "deploy-a")
    except ValueError as error:
        assert "permutation" in str(error)
    else:
        raise AssertionError("candidate mismatch must fail")


def test_naive_proxy_is_sensitive_to_cross_group_cooccurrence():
    signals = fixture()
    original = legacy_proxy_rank(signals)
    injected = signals + [
        signal(f"noise-b-{index}", 8 + index, "trace", "error", "other-group",
               "svc-b", f"external-{index}")
        for index in range(4)
    ]
    assert legacy_proxy_rank(injected) != original


def test_evidence_aware_ranker_is_not_affected_by_cross_group_injection():
    signals = fixture()
    injected = signals + [
        signal(f"noise-b-{index}", 8 + index, "trace", "error", "other-group",
               "svc-b", f"external-{index}")
        for index in range(4)
    ]
    assert rank_hypotheses(signals) == rank_hypotheses(injected)
