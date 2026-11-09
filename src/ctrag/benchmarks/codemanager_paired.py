"""Paired comparison of CodeManager against a transparent legacy-style proxy.

Both arms are evaluated over one explicitly materialized candidate universe per
scenario. The proxy models naive artifact co-occurrence and is NOT claimed to be
the historical production CodeManager.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

from ctrag.experiments.codemanager import EvidenceSignal, rank_hypotheses
from ctrag.benchmarks.codemanager_perturbations import fixture, perturbations


ANOMALOUS = {"warning", "error", "failed", "degraded", "timeout"}


def _artifact(signal: EvidenceSignal) -> str | None:
    return signal.artifact or signal.producer_artifact or signal.consumer_artifact


def candidate_universe(signals: Iterable[EvidenceSignal]) -> tuple[str, ...]:
    """Canonical candidate IDs shared by both rankers (not evidence rows)."""
    return tuple(sorted({
        item.event_id for item in signals
        if item.kind in {"code", "config"}
        and item.status.casefold() in ANOMALOUS
        and _artifact(item)
    }))


def legacy_proxy_rank(signals: Iterable[EvidenceSignal]) -> list[str]:
    """Rank anomalous code/config candidates by raw artifact co-occurrence."""
    items = list(signals)
    candidates = candidate_universe(items)
    by_id = {item.event_id: item for item in items}
    ranked: list[tuple[float, str]] = []
    for event_id in candidates:
        candidate = by_id[event_id]
        artifact = _artifact(candidate)
        votes = sum(
            1 for signal in items
            if signal.event_id != event_id
            and signal.status.casefold() in ANOMALOUS
            and artifact in {
                signal.artifact, signal.producer_artifact, signal.consumer_artifact,
                signal.uses_artifact, signal.attributes.get("artifact"),
                signal.attributes.get("uses"), signal.attributes.get("producer"),
                signal.attributes.get("consumer"),
            }
        )
        ranked.append((float(votes), event_id))
    return [event_id for _, event_id in sorted(ranked, key=lambda row: (-row[0], row[1]))]


def _metrics(ranking: list[str], universe: tuple[str, ...], gold: str) -> dict:
    # Fail closed if a ranker silently drops or invents candidates.
    if len(ranking) != len(set(ranking)) or set(ranking) != set(universe):
        raise ValueError("ranker output must be a permutation of the shared candidate universe")
    rank = ranking.index(gold) + 1 if gold in universe else None
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
        universe = candidate_universe(signals)
        legacy = legacy_proxy_rank(signals)
        modern = [item.event_id for item in rank_hypotheses(signals)]
        rows[name] = {
            "candidate_universe": list(universe),
            "candidate_count": len(universe),
            "legacy_proxy": _metrics(legacy, universe, "deploy-a"),
            "evidence_aware": _metrics(modern, universe, "deploy-a"),
            "legacy_order": legacy,
            "evidence_aware_order": modern,
        }
    metric_names = ("gold_in_candidates", "top1", "recall_at_1", "recall_at_2", "mrr")
    aggregate = {
        arm: {
            metric: sum(float(row[arm][metric]) for row in rows.values()) / len(rows)
            for metric in metric_names
        }
        for arm in ("legacy_proxy", "evidence_aware")
    }
    return {
        "protocol": "codemanager-paired-shared-candidates-v2",
        "baseline_definition": "raw anomalous artifact co-occurrence; no group/time/source deduplication",
        "baseline_is_historical_implementation": False,
        "candidate_parity": "enforced: each ranking must be a permutation of the exact per-scenario candidate IDs",
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
            "Candidate coverage is reported separately; absent gold candidates remain misses in all rank metrics.",
        ],
    }


def main() -> None:
    print(json.dumps(run_paired(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
