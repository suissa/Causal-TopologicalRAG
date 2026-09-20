from __future__ import annotations

from pathlib import Path

from ctrag.benchmarks.early_degradation_robustness import (
    bootstrap_mean_ci,
    controlled_cohort_bootstrap,
    null_stress,
    run,
    sensitivity_grid,
)


def test_sensitivity_grid_covers_requested_parameter_space() -> None:
    rows = sensitivity_grid()
    assert len(rows) == 28
    assert {row["sustained_windows"] for row in rows} == {1, 2, 3, 4}
    assert min(float(row["drift_threshold"]) for row in rows) == 0.05
    assert max(float(row["drift_threshold"]) for row in rows) == 0.20
    assert any(bool(row["positive_lead"]) for row in rows)
    assert any(not bool(row["positive_lead"]) for row in rows)


def test_stationary_null_stress_is_reproducible_and_quiet() -> None:
    result = null_stress(simulations=100, seed=7)
    assert result["simulations"] == 100
    assert result["scenario_false_positive_rate"] <= 0.05
    assert result["post_baseline_observation_days"] == 2500


def test_bootstrap_ci_is_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 2.0, 3.0], iterations=1000, seed=42)
    second = bootstrap_mean_ci([1.0, 2.0, 2.0, 3.0], iterations=1000, seed=42)
    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_controlled_cohort_bootstrap_is_explicitly_synthetic() -> None:
    result = controlled_cohort_bootstrap()
    assert "synthetic" in str(result["scope"])
    assert len(result["lead_times"]) >= 3
    ci = result["bootstrap"]
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_robustness_run_writes_reviewer_artifacts(tmp_path: Path) -> None:
    report = run(tmp_path)
    assert 0.0 <= report["sensitivity"]["positive_lead_fraction"] <= 1.0
    assert (tmp_path / "robustness.json").exists()
    assert (tmp_path / "sensitivity.csv").exists()
    assert (tmp_path / "README.md").exists()
