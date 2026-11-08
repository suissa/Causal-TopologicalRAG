"""Paired synthetic evaluation for CodeManager evidence perturbations.

Run: python -m ctrag.benchmarks.codemanager_perturbations
Outputs JSON to stdout. Scores are heuristic priorities, not probabilities.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Callable

from ctrag.experiments.codemanager import EvidenceSignal, rank_hypotheses

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def signal(i: str, minutes: int, kind: str, status: str, group: str,
           artifact: str | None = None, source: str | None = None, **kw) -> EvidenceSignal:
    return EvidenceSignal(i, T0 + timedelta(minutes=minutes), kind, status,
                         group, artifact=artifact, independent_source=source, **kw)


def fixture() -> list[EvidenceSignal]:
    return [
        signal("deploy-a", 0, "code", "failed", "incident-1", "svc-a",
               expected_version="2", loaded_version="1"),
        signal("deploy-b", 0, "code", "failed", "incident-1", "svc-b"),
        signal("trace-a", 2, "trace", "error", "incident-1", "svc-a", "trace-source"),
        signal("metric-a", 3, "metric", "degraded", "incident-1", "svc-a", "metric-source"),
        signal("log-a", 4, "log", "error", "incident-1", "svc-a", "log-source"),
        signal("healthy-a", 5, "metric", "healthy", "incident-1", "svc-a", "healthy-source"),
        signal("unrelated", 6, "trace", "error", "incident-2", "svc-b", "other-source"),
    ]


def perturbations(base: list[EvidenceSignal]) -> dict[str, list[EvidenceSignal]]:
    by_id = {s.event_id: s for s in base}
    return {
        "identity": list(base),
        "reverse_input": list(reversed(base)),
        "unrelated_noise": base + [signal("noise", 7, "trace", "error", "other", "svc-a", "noise")],
        "same_source_duplicate": base + [signal("trace-a-copy", 2, "trace", "error", "incident-1", "svc-a", "trace-source")],
        "remove_trace_source": [s for s in base if s.event_id != "trace-a"],
        "remove_metric_source": [s for s in base if s.event_id != "metric-a"],
        "remove_log_source": [s for s in base if s.event_id != "log-a"],
        "remove_counter_evidence": [s for s in base if s.event_id != "healthy-a"],
        "cross_group_leak_attempt": base + [signal("cross-group", 8, "metric", "error", "incident-2", "svc-a", "leak")],
    }


def evaluate(signals: list[EvidenceSignal], gold: str = "deploy-a") -> dict:
    ranked = rank_hypotheses(signals)
    ids = [h.event_id for h in ranked]
    present = gold in ids
    rank = ids.index(gold) + 1 if present else None
    return {
        "gold_in_candidates": present,
        "top1": bool(ids and ids[0] == gold),
        "recall_at_1": float(present and rank <= 1),
        "recall_at_2": float(present and rank <= 2),
        "mrr": 1.0 / rank if rank else 0.0,
        "rank": rank,
        "ordered_candidates": ids,
        "scores": {h.event_id: h.score for h in ranked},
    }


def main() -> None:
    base = fixture()
    runs = {name: evaluate(items) for name, items in perturbations(base).items()}
    # Candidate coverage and rank metrics share a fixed scenario denominator.
    summary = {
        "protocol": "codemanager-perturbations-v1",
        "scenario_count": len(runs),
        "denominator": len(runs),
        "aggregate": {
            key: sum(row[key] for row in runs.values()) / len(runs)
            for key in ("gold_in_candidates", "top1", "recall_at_1", "recall_at_2", "mrr")
        },
        "scenarios": runs,
        "limitations": [
            "Synthetic fixture only; no production incidents or human adjudication.",
            "CodeManager scores are heuristic priorities, not calibrated probabilities.",
            "This evaluates rank stability/behavior, not causal identification.",
            "Legacy baseline comparison is not included until its candidate/ranker API is pinned."
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
