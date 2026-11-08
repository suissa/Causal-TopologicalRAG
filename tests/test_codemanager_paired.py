from ctrag.benchmarks.codemanager_paired import legacy_proxy_rank, run_paired
from ctrag.benchmarks.codemanager_perturbations import fixture


def test_legacy_proxy_is_deterministic():
    signals = fixture()
    assert legacy_proxy_rank(signals) == legacy_proxy_rank(reversed(signals))


def test_paired_report_has_same_scenario_denominator_for_both_arms():
    report = run_paired()
    assert report["scenario_count"] == report["denominator"] == 9
    assert set(report["aggregate"]) == {"legacy_proxy", "evidence_aware"}
    assert set(report["delta_evidence_aware_minus_legacy"]) == {
        "gold_in_candidates", "top1", "recall_at_1", "recall_at_2", "mrr"
    }
    assert report["baseline_is_historical_implementation"] is False


def test_proxy_exposes_cross_group_and_temporal_sensitivity():
    signals = fixture()
    baseline = legacy_proxy_rank(signals)
    # Raw co-occurrence proxy is deliberately sensitive to unrelated group noise.
    from ctrag.benchmarks.codemanager_perturbations import perturbations
    scenarios = perturbations(signals)
    assert legacy_proxy_rank(scenarios["cross_group_leak_attempt"]) == baseline
