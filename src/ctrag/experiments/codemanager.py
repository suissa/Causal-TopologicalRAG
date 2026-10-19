"""Evidence-aware CodeManager hypothesis ranking.

This module ranks candidate code/configuration changes without treating a
shared incident identifier or repeated copies of one signal as causal proof.
It is deliberately independent of the commercial experiment's oracle.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping


SUPPORTED_SIGNAL_KINDS = frozenset({"code", "config", "trace", "log", "metric", "symptom"})
CONTRADICTORY_STATUSES = frozenset({"ok", "healthy", "recovered", "resolved"})


@dataclass(frozen=True)
class EvidenceSignal:
    event_id: str
    timestamp: datetime
    kind: str
    status: str
    evidence_group: str
    artifact: str | None = None
    producer_artifact: str | None = None
    consumer_artifact: str | None = None
    uses_artifact: str | None = None
    expected_version: str | None = None
    loaded_version: str | None = None
    expected_contract: str | None = None
    loaded_contract: str | None = None
    independent_source: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedHypothesis:
    event_id: str
    artifact: str
    score: float
    evidence: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    method: str = "artifact_contract_crossing"

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "artifact": self.artifact,
            "score": round(self.score, 4),
            "evidence": list(self.evidence),
            "evidence_kinds": list(self.evidence_kinds),
            "counter_evidence": list(self.counter_evidence),
            "method": self.method,
        }


def _is_anomalous(signal: EvidenceSignal) -> bool:
    return signal.status.casefold() in {"warning", "error", "failed", "degraded", "timeout"}


def _candidate_artifact(signal: EvidenceSignal) -> str | None:
    return signal.artifact or signal.producer_artifact or signal.consumer_artifact


def _artifact_link(candidate: EvidenceSignal, evidence: EvidenceSignal) -> bool:
    artifact = _candidate_artifact(candidate)
    related = {
        evidence.artifact,
        evidence.producer_artifact,
        evidence.consumer_artifact,
        evidence.uses_artifact,
        evidence.attributes.get("artifact"),
        evidence.attributes.get("uses"),
        evidence.attributes.get("producer"),
        evidence.attributes.get("consumer"),
    }
    related.discard(None)
    if artifact and artifact in related:
        return True
    # Explicit consumer/producer linkage is useful even when the evidence names
    # the downstream consumer rather than the changed producer.
    return bool(
        candidate.artifact
        and candidate.artifact == evidence.uses_artifact
        or candidate.producer_artifact
        and candidate.producer_artifact == evidence.consumer_artifact
    )


def _contract_mismatch(signal: EvidenceSignal) -> bool:
    return bool(
        signal.expected_version and signal.loaded_version
        and signal.expected_version != signal.loaded_version
        or signal.expected_contract and signal.loaded_contract
        and signal.expected_contract != signal.loaded_contract
    )


def rank_hypotheses(signals: Iterable[EvidenceSignal]) -> list[RankedHypothesis]:
    """Rank anomalous code/config candidates using distinct evidence sources.

    Scores are bounded heuristic priorities, not calibrated probabilities.
    Duplicate observations from the same source and signal kind count once.
    Evidence from another evidence_group is never joined into a candidate.
    """
    items = sorted(signals, key=lambda item: (item.timestamp, item.event_id))
    candidates = [
        item for item in items
        if item.kind in {"code", "config"} and _is_anomalous(item)
        and _candidate_artifact(item)
    ]
    by_group: dict[str, list[EvidenceSignal]] = defaultdict(list)
    for item in items:
        if item.evidence_group:
            by_group[item.evidence_group].append(item)

    ranked: list[RankedHypothesis] = []
    for candidate in candidates:
        artifact = _candidate_artifact(candidate)
        peers = by_group[candidate.evidence_group]
        support: dict[str, EvidenceSignal] = {}
        counter: dict[str, EvidenceSignal] = {}
        for signal in peers:
            if signal.event_id == candidate.event_id or signal.timestamp < candidate.timestamp:
                continue
            if not _artifact_link(candidate, signal):
                continue
            if signal.kind not in {"trace", "log", "metric", "symptom"}:
                continue
            key = signal.independent_source or signal.kind
            if _is_anomalous(signal):
                support.setdefault(key, signal)
            elif signal.status.casefold() in CONTRADICTORY_STATUSES:
                counter.setdefault(key, signal)

        kinds = {signal.kind for signal in support.values()}
        score = 0.25
        score += min(0.40, 0.12 * len(support))
        score += 0.15 if _contract_mismatch(candidate) else 0.0
        score += 0.10 if candidate.kind == "code" else 0.05
        score -= min(0.25, 0.10 * len(counter))
        # Multiple signal kinds provide more diversity than repeated samples.
        score += min(0.10, max(0, len(kinds) - 1) * 0.025)
        score = max(0.0, min(0.99, score))
        ranked.append(RankedHypothesis(
            event_id=candidate.event_id,
            artifact=str(artifact),
            score=score,
            evidence=tuple(sorted(signal.event_id for signal in support.values())),
            evidence_kinds=tuple(sorted(kinds)),
            counter_evidence=tuple(sorted(signal.event_id for signal in counter.values())),
        ))

    return sorted(ranked, key=lambda item: (-item.score, item.event_id))
