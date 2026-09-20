from __future__ import annotations

from ctrag.benchmarks.eroded_path_rescue import ErodedPathConfig, run


def test_eroded_path_rescue_discriminates_high_success_from_eroded_distractor(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)

    outcomes = report["outcomes"]
    metrics = report["metrics"]
    rankings = report["rankings"]
    components = report["ctrag_components"]

    assert outcomes["provider_path_eroded_vs_failure"] is True
    assert outcomes["manual_path_eroded_vs_failure"] is True
    assert outcomes["provider_history_preserved"] is True
    assert outcomes["manual_history_preserved"] is True

    assert outcomes["provider_success_rate"] == 1.0
    assert outcomes["manual_success_rate"] == 0.1

    # Recency is deliberately fooled by the newer low-success manual recovery.
    assert outcomes["recency_prefers_low_success_distractor"] is True
    assert rankings["recency_recovered_only"][0] == "manual_recovered"

    # Structural reachability is a ceiling/candidate-set baseline, not a ranker.
    assert outcomes["structural_baseline_contains_both_recoveries"] is True
    assert set(rankings["structural_recovered_unranked"]) == {
        "provider_recovered",
        "manual_recovered",
    }

    # CT-RAG must discriminate rather than simply rescue every eroded path.
    assert outcomes["ctrag_prefers_high_success_eroded_path"] is True
    assert outcomes["ctrag_does_not_false_rescue_manual_at_1"] is True
    assert rankings["ctrag_recovery"][0] == "provider_recovered"
    assert metrics["ctrag_precision_at_1"] == 1.0
    assert metrics["ctrag_false_rescue_at_1"] == 0.0

    assert components["provider_recovered"]["historical_success_rate"] > components["manual_recovered"]["historical_success_rate"]
    assert components["provider_recovered"]["historical_support"] > 0.0
    assert components["manual_recovered"]["historical_support"] > 0.0


def test_eroded_path_rescue_uses_stronger_baselines(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    metrics = report["metrics"]

    assert metrics["terrain_only_precision_at_1"] == 0.0
    assert metrics["recency_precision_at_1"] == 0.0
    assert metrics["structural_precision_at_2"] == 0.5
    assert metrics["ctrag_precision_at_1"] == 1.0


def test_eroded_path_rescue_writes_machine_readable_artifacts(tmp_path) -> None:
    run(ErodedPathConfig(), tmp_path)

    assert (tmp_path / "eroded-path-rescue.json").exists()
    assert (tmp_path / "eroded-path-rescue.csv").exists()
