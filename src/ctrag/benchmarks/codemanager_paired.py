"""Paired comparison of CodeManager against a transparent legacy-style proxy.

The proxy intentionally models naive artifact co-occurrence: anomalous signals
sharing an artifact string add votes, without evidence-group, temporal, or
independent-source safeguards. It is NOT claimed to be the historical
production CodeManager; use it only as a controlled reference arm.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Iterable

from ctrag.experiments.codemanager import EvidenceSignal, rank_hypotheses
from ctrag.benchmarks.codemanager_perturbations import fixture, perturbations


def _artifact(signal: EvidenceSignal) -> str | None:
    return signal.artifact or signal.producer_artifact or signal.consumer_artifact


def legacy_proxy_rank(signals: Iterable[EvidenceSignal]) -> list[str]:
    """Rank anomalous code/config candidates by raw artifact co-occurrence."""
    items = list(signals)
    candidates = [
        item for item in items
        if item.kind in {"code", "config"}
        and item.status.casefold() in {"warning", "error", "failed", "degraded", "timeout"}
        and _artifact(item)
    ]
    ranked: list[tuple[float, str]] = []
    for candidate in candidates:
        artifact = _artifact(candidate)
        votes = sum(
            1 for signal in items
            if signal.event_id != candidate.event_id
            and signal.status.casefold() in {"warning", "error", "failed", "degraded", "timeout"}
            and artifact in {
                signal.artifact, signal.producer_artifact, signal.consumer_artifact,
                signal.uses_artifact, signal.attributes.get("artifact"),
                signal.attributes.get("uses"), signal.attributes.get("producer"),
                signal.attributes.get("consumer"),
            }
        )
        ranked.append((float(votes), candidate.event_id))
    return [event_id for _, event_id in sorted(ranked, key=lambda row: (-row[0], row[1]))]


def _metrics(ranking: list[str], gold: str) -> dict[str, float | bool | int | None]:
    rank = ranking.index(gold) + 1 if gold in ranking else None
    return {
        "gold_in_candidates": rank is not None,
        "top1": bool(ranking and ranking[0] == gold),
        "recall_at_1": float(rank == 1),
        "recall_at_2": float(rank is not None and rank <= 2),
        "mrr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


def run_paired() -> dict:
    scenarios = perturbations(fixture())
    rows = {}
    for name, signals in scenarios.items():
        modern = [item.event_id for item in rank_hypotheses(signals)]
        legacy = legacy_proxy_rank(signals)
        rows[name] = {
            "legacy_proxy": _metrics(legacy, "deploy-a"),
            "evidence_aware": _metrics(modern, "deploy-a"),
            "legacy_order": legacy,
            "evidence_aware_order": modern,
        }
    metric_names = ("gold_in_candidates", "top1", "recall_at_1", "recall_at_2", "mrr")
    aggregate = {}
    for arm in ("legacy_proxy", "evidence_aware"):
        aggregate[arm] = {
            metric: sum(float(row[arm][metric]) for row in rows.values()) / len(rows)
            for metric in metric_names
        }
    return {
        "protocol": "codemanager-paired-proxy-v1",
        "baseline_definition": "raw anomalous artifact co-occurrence; no group/time/source deduplication",
        "baseline_is_historical_implementation": False,
        "scenario_count": len(rows),
        "denominator": len(rows),
        "aggregate": aggregate,
        "delta_evidence_aware_minus_legacy": {
            metric: aggregate["evidence_aware"][metric] - aggregate["legacy_proxy"][metric]
            for metric in metric_names
        },
        "scenarios": rows,
        "limitations": [
            "Legacy arm is an explicit naive proxy, not verified historical production code.",
            "Synthetic fixture only; no production incidents or human adjudication.",
            "Heuristic scores are not calibrated probabilities and rankings do not prove causality.",
        ],
    }


def main() -> None:
    print(json.dumps(run_paired(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
