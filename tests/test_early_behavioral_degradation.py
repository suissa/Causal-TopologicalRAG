from __future__ import annotations

from pathlib import Path

from ctrag.benchmarks.early_behavioral_degradation import (
    EarlyDegradationConfig,
    default_windows,
    run,
    total_variation,
)


def test_total_variation_on_two_attractor_distributions() -> None:
    healthy = {"Recovered": 0.9, "HumanIntervention": 0.1}
    shifted = {"Recovered": 0.7, "HumanIntervention": 0.3}

    assert total_variation(healthy, shifted) == 0.2


def test_early_behavioral_degradation_precedes_infrastructure_alert(tmp_path: Path) -> None:
    report = run(EarlyDegradationConfig(), tmp_path)

    outcomes = report["outcomes"]
    assert outcomes["behavioral_alert_day"] == 8
    assert outcomes["infrastructure_alert_day"] == 10
    assert outcomes["lead_time_days"] == 2
    assert outcomes["behavioral_alert_precedes_infrastructure"] is True
    assert outcomes["baseline_false_alerts"] == 0

    rows = report["rows"]
    assert rows[6]["day"] == 7
    assert rows[6]["basin_drift_tv"] == 0.13
    assert rows[7]["day"] == 8
    assert rows[7]["basin_drift_tv"] == 0.2
    assert rows[7]["behavioral_alert"] is True

    assert (tmp_path / "early-behavioral-degradation.json").exists()
    assert (tmp_path / "early-behavioral-degradation.csv").exists()
    assert (tmp_path / "manifest.json").exists()


def test_no_behavioral_alert_without_sustained_drift(tmp_path: Path) -> None:
    windows = default_windows()[:7]
    report = run(EarlyDegradationConfig(sustained_windows=2), tmp_path, windows=windows)

    assert report["outcomes"]["behavioral_alert_day"] is None
    assert report["outcomes"]["infrastructure_alert_day"] is None
    assert report["outcomes"]["lead_time_days"] is None
