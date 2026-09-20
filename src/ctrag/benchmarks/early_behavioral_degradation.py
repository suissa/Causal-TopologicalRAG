"""Deterministic early behavioral-degradation experiment.

Question:
Can basin drift detect a production-behavior change before a traditional
infrastructure threshold fires?

This experiment is deliberately controlled and synthetic. It establishes the
measurement protocol and lead-time metric; it is not evidence that the same
lead time exists in arbitrary production systems.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EarlyDegradationConfig:
    baseline_days: int = 5
    drift_threshold: float = 0.10
    sustained_windows: int = 2
    infrastructure_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.baseline_days < 1:
            raise ValueError("baseline_days must be positive")
        if not 0.0 <= self.drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be within [0, 1]")
        if self.sustained_windows < 1:
            raise ValueError("sustained_windows must be positive")
        if not 0.0 <= self.infrastructure_threshold <= 1.0:
            raise ValueError("infrastructure_threshold must be within [0, 1]")


@dataclass(frozen=True)
class DailyWindow:
    day: int
    retry_recovered: int
    timeout_human: int
    infrastructure_metric: float

    @property
    def total(self) -> int:
        return self.retry_recovered + self.timeout_human

    def absorption_distribution(self) -> dict[str, float]:
        if self.total <= 0:
            raise ValueError("daily window must contain observations")
        return {
            "Recovered": self.retry_recovered / self.total,
            "HumanIntervention": self.timeout_human / self.total,
        }


def default_windows() -> tuple[DailyWindow, ...]:
    """Controlled degradation: behavior shifts before infrastructure saturation."""
    return (
        DailyWindow(1, 90, 10, 0.55),
        DailyWindow(2, 90, 10, 0.56),
        DailyWindow(3, 90, 10, 0.57),
        DailyWindow(4, 90, 10, 0.58),
        DailyWindow(5, 90, 10, 0.59),
        DailyWindow(6, 84, 16, 0.62),
        DailyWindow(7, 77, 23, 0.68),
        DailyWindow(8, 70, 30, 0.73),
        DailyWindow(9, 60, 40, 0.78),
        DailyWindow(10, 50, 50, 0.86),
        DailyWindow(11, 42, 58, 0.91),
    )


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def aggregate_distribution(windows: tuple[DailyWindow, ...]) -> dict[str, float]:
    recovered = sum(window.retry_recovered for window in windows)
    human = sum(window.timeout_human for window in windows)
    total = recovered + human
    if total <= 0:
        raise ValueError("baseline must contain observations")
    return {
        "Recovered": recovered / total,
        "HumanIntervention": human / total,
    }


def run(
    config: EarlyDegradationConfig,
    output: Path,
    *,
    windows: tuple[DailyWindow, ...] | None = None,
) -> dict:
    windows = windows or default_windows()
    if len(windows) <= config.baseline_days:
        raise ValueError("experiment requires at least one post-baseline window")

    output.mkdir(parents=True, exist_ok=True)
    baseline = aggregate_distribution(windows[: config.baseline_days])

    rows: list[dict[str, object]] = []
    sustained = 0
    behavioral_alert_day: int | None = None
    infrastructure_alert_day: int | None = None

    for window in windows:
        distribution = window.absorption_distribution()
        drift = total_variation(baseline, distribution)

        if window.day > config.baseline_days and drift >= config.drift_threshold:
            sustained += 1
        else:
            sustained = 0

        behavioral_alert = (
            window.day > config.baseline_days
            and sustained >= config.sustained_windows
        )
        infrastructure_alert = window.infrastructure_metric >= config.infrastructure_threshold

        if behavioral_alert_day is None and behavioral_alert:
            behavioral_alert_day = window.day
        if infrastructure_alert_day is None and infrastructure_alert:
            infrastructure_alert_day = window.day

        rows.append({
            "day": window.day,
            "retry_recovered": window.retry_recovered,
            "timeout_human": window.timeout_human,
            "recovered_probability": distribution["Recovered"],
            "human_intervention_probability": distribution["HumanIntervention"],
            "basin_drift_tv": drift,
            "drift_threshold_exceeded": drift >= config.drift_threshold,
            "sustained_drift_windows": sustained,
            "behavioral_alert": behavioral_alert,
            "infrastructure_metric": window.infrastructure_metric,
            "infrastructure_alert": infrastructure_alert,
        })

    lead_time_days = None
    if behavioral_alert_day is not None and infrastructure_alert_day is not None:
        lead_time_days = infrastructure_alert_day - behavioral_alert_day

    report = {
        "schema_version": 1,
        "claim_scope": (
            "controlled synthetic early-warning mechanism; positive lead time here "
            "does not establish production forecasting performance"
        ),
        "question": (
            "Can absorption-distribution basin drift detect behavioral degradation "
            "before a traditional infrastructure threshold fires?"
        ),
        "config": asdict(config),
        "baseline_absorption_distribution": baseline,
        "rows": rows,
        "outcomes": {
            "behavioral_alert_day": behavioral_alert_day,
            "infrastructure_alert_day": infrastructure_alert_day,
            "lead_time_days": lead_time_days,
            "behavioral_alert_precedes_infrastructure": (
                lead_time_days is not None and lead_time_days > 0
            ),
            "baseline_false_alerts": sum(
                1 for row in rows
                if int(row["day"]) <= config.baseline_days and bool(row["behavioral_alert"])
            ),
        },
        "protocol": {
            "basin_definition": (
                "Each daily operation window ends in one of two observed attractors: "
                "Recovered or HumanIntervention."
            ),
            "drift_estimator": (
                "Total-variation divergence between the healthy baseline absorption "
                "distribution and each chronological daily window."
            ),
            "detector": (
                f"Alert after {config.sustained_windows} consecutive post-baseline "
                f"windows with TV drift >= {config.drift_threshold:.3f}."
            ),
            "traditional_alert": (
                "Independent infrastructure metric threshold; it does not use basin labels."
            ),
            "future_leakage": (
                "Each row is evaluated chronologically using a baseline fixed before "
                "the degradation period and the current/past window only."
            ),
        },
    }

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "early-behavioral-degradation.json").write_text(
        encoded, encoding="utf-8", newline="\n"
    )

    with (output / "early-behavioral-degradation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        "# Early Behavioral Degradation — Controlled Experiment",
        "",
        f"- Behavioral basin-drift alert: day {behavioral_alert_day}",
        f"- Traditional infrastructure alert: day {infrastructure_alert_day}",
        f"- Lead time: {lead_time_days} day(s)",
        f"- Baseline false alerts: {report['outcomes']['baseline_false_alerts']}",
        "",
        "This is a deterministic mechanism test, not a production forecasting claim.",
        "",
    ]
    (output / "README.md").write_text("\n".join(summary), encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": 1,
        "command": "python -m ctrag.benchmarks.early_behavioral_degradation",
        "results_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic early behavioral-degradation experiment"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/early-behavioral-degradation"),
    )
    args = parser.parse_args(argv)
    report = run(EarlyDegradationConfig(), args.output)
    outcomes = report["outcomes"]
    print(
        "behavioral_alert_day=",
        outcomes["behavioral_alert_day"],
        " infrastructure_alert_day=",
        outcomes["infrastructure_alert_day"],
        " lead_time_days=",
        outcomes["lead_time_days"],
        sep="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
