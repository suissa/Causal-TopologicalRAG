from __future__ import annotations

from ctrag.benchmarks.eroded_path_rescue import ErodedPathConfig, run


def test_raw_success_baseline_is_fooled_by_unreachable_high_rate_path(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    rankings = report["rankings"]
    outcomes = report["outcomes"]
    metrics = report["metrics"]

    assert outcomes["raw_success_prefers_unreachable_path"] is True
    assert rankings["raw_success_global"][0] == "global_recovered"
    assert outcomes["global_high_success_is_unreachable_from_retry"] is True
    assert outcomes["ctrag_excludes_unreachable_high_success_path"] is True
    assert metrics["raw_success_precision_at_1"] == 0.0


def test_ctrag_prefers_reachable_high_quality_eroded_recovery(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    rankings = report["rankings"]
    metrics = report["metrics"]

    assert report["outcomes"]["ctrag_prefers_provider"] is True
    assert rankings["ctrag_recovery"][0] == "provider_recovered"
    assert metrics["ctrag_precision_at_1"] == 1.0
    assert metrics["terrain_only_precision_at_1"] == 0.0
    assert metrics["recency_precision_at_1"] == 0.0


def test_false_rescue_is_reported_beyond_top1(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    metrics = report["metrics"]

    assert metrics["false_rescue_at_1"] == 0.0
    assert 0.0 <= metrics["false_rescue_at_2"] <= 1.0
    assert 0.0 <= metrics["false_rescue_at_3"] <= 1.0
    assert 0.0 <= metrics["false_rescue_at_4"] <= 1.0
    assert metrics["false_rescue_at_2"] >= metrics["false_rescue_at_1"]


def test_structural_baseline_is_reachability_ceiling_not_global_ranker(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    structural = report["rankings"]["structural_recovered_unranked"]

    assert "provider_recovered" in structural
    assert "manual_recovered" in structural
    assert "script_recovered" in structural
    assert "cache_recovered" in structural
    assert "global_recovered" not in structural


def test_eroded_path_rescue_writes_machine_readable_artifacts(tmp_path) -> None:
    run(ErodedPathConfig(), tmp_path)
    assert (tmp_path / "eroded-path-rescue.json").exists()
    assert (tmp_path / "eroded-path-rescue.csv").exists()
