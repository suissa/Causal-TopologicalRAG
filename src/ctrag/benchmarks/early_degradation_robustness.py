"""Robustness analyses for the early behavioral-degradation mechanism.

Contains three reviewer-facing checks:
1. sensitivity grid over drift threshold and sustained-window count;
2. stationary healthy null scenarios for false-positive stress;
3. deterministic bootstrap confidence intervals for a controlled incident cohort.

All generated cohorts are synthetic mechanism tests. They must not be reported as
production confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from .early_behavioral_degradation import (
    DailyWindow,
    EarlyDegradationConfig,
    aggregate_distribution,
    default_windows,
    total_variation,
)


@dataclass(frozen=True)
class Detection:
    behavioral_alert_day: int | None
    infrastructure_alert_day: int | None
    lead_time_days: int | None


def detect(windows: tuple[DailyWindow, ...], config: EarlyDegradationConfig) -> Detection:
    baseline = aggregate_distribution(windows[: config.baseline_days])
    sustained = 0
    behavioral: int | None = None
    infrastructure: int | None = None

    for window in windows:
        drift = total_variation(baseline, window.absorption_distribution())
        if window.day > config.baseline_days and drift >= config.drift_threshold:
            sustained += 1
        else:
            sustained = 0
        if (
            behavioral is None
            and window.day > config.baseline_days
            and sustained >= config.sustained_windows
        ):
            behavioral = window.day
        if infrastructure is None and window.infrastructure_metric >= config.infrastructure_threshold:
            infrastructure = window.day

    lead = None
    if behavioral is not None and infrastructure is not None:
        lead = infrastructure - behavioral
    return Detection(behavioral, infrastructure, lead)


def sensitivity_grid() -> list[dict[str, object]]:
    thresholds = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
    sustained_values = (1, 2, 3, 4)
    windows = default_windows()
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        for sustained in sustained_values:
            config = EarlyDegradationConfig(
                drift_threshold=threshold,
                sustained_windows=sustained,
            )
            result = detect(windows, config)
            rows.append({
                "drift_threshold": threshold,
                "sustained_windows": sustained,
                "behavioral_alert_day": result.behavioral_alert_day,
                "infrastructure_alert_day": result.infrastructure_alert_day,
                "lead_time_days": result.lead_time_days,
                "positive_lead": result.lead_time_days is not None and result.lead_time_days > 0,
            })
    return rows


def stationary_null_windows(
    *,
    seed: int,
    days: int = 30,
    baseline_days: int = 5,
) -> tuple[DailyWindow, ...]:
    rng = random.Random(seed)
    result: list[DailyWindow] = []
    for day in range(1, days + 1):
        # Healthy stochastic variation around a 90/10 attractor split.
        recovered = max(84, min(96, int(round(rng.gauss(90.0, 2.0)))))
        human = 100 - recovered
        # Independent infrastructure noise safely below the production alert threshold.
        infrastructure = max(0.40, min(0.78, rng.gauss(0.58, 0.045)))
        result.append(DailyWindow(day, recovered, human, infrastructure))
    if len(result) <= baseline_days:
        raise ValueError("null scenario requires post-baseline windows")
    return tuple(result)


def null_stress(
    *,
    simulations: int = 500,
    days: int = 30,
    seed: int = 20260920,
    config: EarlyDegradationConfig | None = None,
) -> dict[str, object]:
    config = config or EarlyDegradationConfig()
    false_alarm_days: list[int] = []
    observation_days = 0
    for index in range(simulations):
        windows = stationary_null_windows(seed=seed + index, days=days, baseline_days=config.baseline_days)
        result = detect(windows, config)
        observation_days += days - config.baseline_days
        if result.behavioral_alert_day is not None:
            false_alarm_days.append(result.behavioral_alert_day)

    false_alarms = len(false_alarm_days)
    fpr = false_alarms / simulations
    mtbf = None if false_alarms == 0 else observation_days / false_alarms
    return {
        "simulations": simulations,
        "days_per_simulation": days,
        "post_baseline_observation_days": observation_days,
        "false_alarm_scenarios": false_alarms,
        "scenario_false_positive_rate": fpr,
        "mean_time_between_false_alarms_days": mtbf,
        "mtbf_lower_bound_days_if_zero_false_alarms": observation_days if false_alarms == 0 else None,
        "first_false_alarm_days": false_alarm_days[:20],
    }


def controlled_incident_cohort() -> tuple[tuple[DailyWindow, ...], ...]:
    """Generate mechanism variants; these are not production incidents."""
    base = default_windows()
    variants: list[tuple[DailyWindow, ...]] = []
    # Vary degradation timing and infrastructure response independently.
    for shift in (0, 1, 2):
        rows: list[DailyWindow] = []
        for item in base:
            day = item.day
            if day <= 5 + shift:
                recovered = 90
                human = 10
            else:
                idx = min(len(base) - 1, max(5, day - shift - 1))
                source = base[idx]
                recovered = source.retry_recovered
                human = source.timeout_human
            infra = max(0.0, min(1.0, item.infrastructure_metric + (0.015 * shift)))
            rows.append(DailyWindow(day, recovered, human, infra))
        variants.append(tuple(rows))
    for acceleration in (1.05, 1.10, 1.15):
        rows = []
        for item in base:
            human = 10 if item.day <= 5 else min(70, int(round(item.timeout_human * acceleration)))
            recovered = 100 - human
            infra = max(0.0, min(1.0, item.infrastructure_metric - 0.02))
            rows.append(DailyWindow(item.day, recovered, human, infra))
        variants.append(tuple(rows))
    return tuple(variants)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = 1000,
    seed: int = 20260920,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(iterations):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": n,
        "iterations": iterations,
        "confidence": confidence,
        "mean": mean(values),
        "median": median(values),
        "ci_low": percentile(samples, alpha),
        "ci_high": percentile(samples, 1.0 - alpha),
    }


def temporal_drift_detection_accuracy(
    alert_days: list[int | None],
    onset_days: list[int | None],
    *,
    tolerance_days: int = 3,
) -> dict[str, object]:
    """Score labeled drift detection with explicit misses and false alarms."""
    if len(alert_days) != len(onset_days):
        raise ValueError("alert_days and onset_days must have equal length")
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be non-negative")
    correct = 0
    delays: list[int] = []
    false_positives = 0
    false_negatives = 0
    for alert, onset in zip(alert_days, onset_days):
        if onset is None:
            if alert is None:
                correct += 1
            else:
                false_positives += 1
            continue
        if alert is None:
            false_negatives += 1
            continue
        delay = alert - onset
        delays.append(delay)
        if abs(delay) <= tolerance_days:
            correct += 1
    n = len(alert_days)
    return {
        "n": n,
        "tolerance_days": tolerance_days,
        "accuracy": 0.0 if n == 0 else correct / n,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detection_delays": delays,
        "mean_detection_delay": None if not delays else mean(delays),
    }

def controlled_cohort_bootstrap() -> dict[str, object]:
    config = EarlyDegradationConfig()
    cohort = controlled_incident_cohort()
    detections = [detect(windows, config) for windows in cohort]
    leads = [float(item.lead_time_days) for item in detections if item.lead_time_days is not None]
    onset_days: list[int | None] = []
    for windows in cohort:
        baseline = aggregate_distribution(windows[: config.baseline_days])
        baseline_human = baseline["HumanIntervention"]
        onset = next(
            (
                window.day
                for window in windows[config.baseline_days :]
                if window.absorption_distribution()["HumanIntervention"] > baseline_human
            ),
            None,
        )
        onset_days.append(onset)
    detection_metric = temporal_drift_detection_accuracy(
        [item.behavioral_alert_day for item in detections],
        onset_days,
        tolerance_days=3,
    )
    return {
        "scope": "synthetic controlled incident variants; not a production CI",
        "lead_times": leads,
        "bootstrap": bootstrap_mean_ci(leads, iterations=1000),
        "temporal_drift_detection": detection_metric,
    }


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    sensitivity = sensitivity_grid()
    null = null_stress()
    bootstrap = controlled_cohort_bootstrap()

    positive = sum(1 for row in sensitivity if row["positive_lead"])
    report = {
        "schema_version": 1,
        "claim_scope": "synthetic robustness analysis only",
        "sensitivity": {
            "rows": sensitivity,
            "positive_lead_configurations": positive,
            "total_configurations": len(sensitivity),
            "positive_lead_fraction": positive / len(sensitivity),
        },
        "null_stress": null,
        "bootstrap": bootstrap,
    }
    (output / "robustness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "sensitivity.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity[0]))
        writer.writeheader()
        writer.writerows(sensitivity)
    (output / "README.md").write_text(
        "# Early-degradation robustness\n\n"
        f"- Sensitivity configurations with positive lead: {positive}/{len(sensitivity)}\n"
        f"- Null scenario false-positive rate: {null['scenario_false_positive_rate']:.4f}\n"
        f"- Controlled-cohort bootstrap mean lead: {bootstrap['bootstrap']['mean']:.3f} days\n"
        f"- Controlled-cohort 95% bootstrap CI: "
        f"[{bootstrap['bootstrap']['ci_low']:.3f}, {bootstrap['bootstrap']['ci_high']:.3f}] days\n\n"
        "All values are synthetic mechanism/stress results, not production estimates.\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CT-RAG early-warning robustness analyses")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/early-degradation-robustness"),
    )
    args = parser.parse_args(argv)
    report = run(args.output)
    print(json.dumps({
        "positive_lead_fraction": report["sensitivity"]["positive_lead_fraction"],
        "null_fpr": report["null_stress"]["scenario_false_positive_rate"],
        "bootstrap": report["bootstrap"]["bootstrap"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
