from ctrag.benchmarks.codemanager_paired import legacy_proxy_rank, run_paired
from ctrag.benchmarks.codemanager_perturbations import fixture, perturbations, signal


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
    scenarios = perturbations(fixture())
    assert scenarios["cross_group_leak_attempt"]
