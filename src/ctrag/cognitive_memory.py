"""Epistemically separated cognitive memory contract and SQLite reference adapter.

This module stores interpretations about experience. It never projects graph
adjacency into causal relations and never mutates the authoritative CTEG.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


def _identity(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _aware(value: datetime, label: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


class CognitiveNodeKind(str, Enum):
    SUBJECT = "Subject"
    OBSERVATION = "Observation"
    SIGNAL_VECTOR = "SignalVector"
    BEHAVIOR_STATE = "BehaviorState"
    BEHAVIOR_ID = "BehaviorID"
    COG_TRANSITION = "CogTransition"
    SITUATION = "Situation"
    INTENT = "Intent"
    ENTITY = "Entity"
    EVIDENCE = "Evidence"
    PROSODY_BASELINE = "ProsodyBaseline"
    TEXT_BASELINE = "TextBaseline"
    SITUATIONAL_PATTERN = "SituationalPattern"
    COGNITIVE_EMOTIONAL_HYPOTHESIS = "CognitiveEmotionalHypothesis"
    SNAPSHOT = "Snapshot"


class CognitiveRelation(str, Enum):
    HAS_BASELINE = "HAS_BASELINE"
    HAS_SNAPSHOT = "HAS_SNAPSHOT"
    OBSERVED_FOR = "OBSERVED_FOR"
    EMITS_SIGNAL = "EMITS_SIGNAL"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    FROM_STATE = "FROM_STATE"
    VIA_TRANSITION = "VIA_TRANSITION"
    TO_STATE = "TO_STATE"
    OBSERVED_IN = "OBSERVED_IN"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    CONSOLIDATES = "CONSOLIDATES"
    CONTAINS_HYPOTHESIS = "CONTAINS_HYPOTHESIS"


@dataclass(frozen=True, slots=True)
class CognitiveNode:
    id: str
    kind: CognitiveNodeKind
    subject_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identity(self.id, "node id")
        _identity(self.subject_id, "subject_id")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")
        json.dumps(self.payload, sort_keys=True, default=_json_default)


@dataclass(frozen=True, slots=True)
class CognitiveEdge:
    source_id: str
    target_id: str
    relation: CognitiveRelation
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        _identity(self.source_id, "source_id")
        _identity(self.target_id, "target_id")
        if self.evidence_ref is not None:
            _identity(self.evidence_ref, "evidence_ref")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    id: str
    source_ref: str
    source_kind: str
    observed_at: datetime
    polarity: str

    def __post_init__(self) -> None:
        for name in ("id", "source_ref", "source_kind"):
            _identity(getattr(self, name), name)
        _aware(self.observed_at, "observed_at")
        if self.polarity not in {"supports", "contradicts"}:
            raise ValueError("polarity must be 'supports' or 'contradicts'")


@dataclass(frozen=True, slots=True)
class CognitiveEmotionalHypothesis:
    id: str
    subject_id: str
    statement: str
    confidence: float
    supporting_evidence: tuple[EvidenceRef, ...]
    counter_evidence: tuple[EvidenceRef, ...]
    valid_from: datetime
    valid_until: datetime | None
    source_signal_refs: tuple[str, ...]
    transition_refs: tuple[str, ...]
    pattern_refs: tuple[str, ...]
    inference_method: str
    model_version: str
    skill_version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _identity(self.id, "hypothesis id")
        _identity(self.subject_id, "subject_id")
        _identity(self.statement, "statement")
        for name in ("inference_method", "model_version", "skill_version"):
            _identity(getattr(self, name), name)
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        _aware(self.valid_from, "valid_from")
        _aware(self.created_at, "created_at")
        if self.valid_until is not None:
            _aware(self.valid_until, "valid_until")
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until cannot precede valid_from")
        if any(ref.polarity != "supports" for ref in self.supporting_evidence):
            raise ValueError("supporting_evidence must contain supports refs only")
        if any(ref.polarity != "contradicts" for ref in self.counter_evidence):
            raise ValueError("counter_evidence must contain contradicts refs only")
        ids = [ref.id for ref in self.supporting_evidence + self.counter_evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence refs must be unique across both polarities")
        for name in ("source_signal_refs", "transition_refs", "pattern_refs"):
            if any(not isinstance(item, str) or not item.strip() for item in getattr(self, name)):
                raise ValueError(f"{name} entries must be non-empty strings")

    def is_valid_at(self, instant: datetime) -> bool:
        _aware(instant, "instant")
        return self.valid_from <= instant and (
            self.valid_until is None or instant <= self.valid_until
        )


@dataclass(frozen=True, slots=True)
class BehaviorSnapshot:
    id: str
    subject_id: str
    created_at: datetime
    behavior_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    baseline_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.id, "snapshot id")
        _identity(self.subject_id, "subject_id")
        _aware(self.created_at, "created_at")
        for values in (self.behavior_ids, self.hypothesis_ids, self.baseline_refs):
            if any(not value.strip() for value in values):
                raise ValueError("snapshot references must be non-empty strings")


class CognitiveMemoryStore(Protocol):
    def put_node(self, node: CognitiveNode) -> None: ...
    def get_node(self, node_id: str) -> CognitiveNode | None: ...
    def add_edge(self, edge: CognitiveEdge) -> None: ...
    def put_hypothesis(self, hypothesis: CognitiveEmotionalHypothesis) -> None: ...
    def get_hypotheses(self, subject_id: str, *, valid_at: datetime | None = None) -> list[CognitiveEmotionalHypothesis]: ...
    def put_snapshot(self, snapshot: BehaviorSnapshot) -> None: ...
    def get_snapshot(self, snapshot_id: str) -> BehaviorSnapshot | None: ...
    def get_baselines(self, subject_id: str) -> list[CognitiveNode]: ...
    def get_hypothesis_evidence(self, hypothesis_id: str) -> tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]: ...


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _hypothesis_dict(item: CognitiveEmotionalHypothesis) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(item), default=_json_default))


def _hypothesis_from(raw: dict[str, Any]) -> CognitiveEmotionalHypothesis:
    def dt(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    def refs(values: list[dict[str, Any]]) -> tuple[EvidenceRef, ...]:
        return tuple(EvidenceRef(**{**value, "observed_at": dt(value["observed_at"])}) for value in values)
    return CognitiveEmotionalHypothesis(
        id=raw["id"], subject_id=raw["subject_id"], statement=raw["statement"],
        confidence=raw["confidence"], supporting_evidence=refs(raw["supporting_evidence"]),
        counter_evidence=refs(raw["counter_evidence"]), valid_from=dt(raw["valid_from"]),
        valid_until=None if raw["valid_until"] is None else dt(raw["valid_until"]),
        source_signal_refs=tuple(raw["source_signal_refs"]),
        transition_refs=tuple(raw["transition_refs"]), pattern_refs=tuple(raw["pattern_refs"]),
        inference_method=raw["inference_method"], model_version=raw["model_version"],
        skill_version=raw["skill_version"], created_at=dt(raw["created_at"]),
    )


class SQLiteCognitiveMemoryStore:
    """Reference persistence adapter; CCTDb implementations can satisfy the same protocol."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cognitive_nodes (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cognitive_subject_kind
                    ON cognitive_nodes(subject_id, kind);
                CREATE TABLE IF NOT EXISTS cognitive_edges (
                    source_id TEXT NOT NULL, target_id TEXT NOT NULL,
                    relation TEXT NOT NULL, evidence_ref TEXT,
                    PRIMARY KEY(source_id, target_id, relation)
                );
                CREATE TABLE IF NOT EXISTS cognitive_hypotheses (
                    id TEXT PRIMARY KEY, subject_id TEXT NOT NULL,
                    valid_from TEXT NOT NULL, valid_until TEXT, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_hypothesis_subject_validity
                    ON cognitive_hypotheses(subject_id, valid_from, valid_until);
                CREATE TABLE IF NOT EXISTS behavior_snapshots (
                    id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def put_node(self, node: CognitiveNode) -> None:
        payload = json.dumps(node.payload, sort_keys=True, default=_json_default)
        with self._connect() as db:
            db.execute(
                "INSERT INTO cognitive_nodes VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, "
                "subject_id=excluded.subject_id, payload=excluded.payload",
                (node.id, node.kind.value, node.subject_id, payload),
            )

    def get_node(self, node_id: str) -> CognitiveNode | None:
        with self._connect() as db:
            row = db.execute("SELECT id, kind, subject_id, payload FROM cognitive_nodes WHERE id=?", (node_id,)).fetchone()
        return None if row is None else CognitiveNode(row[0], CognitiveNodeKind(row[1]), row[2], json.loads(row[3]))

    def add_edge(self, edge: CognitiveEdge) -> None:
        with self._connect() as db:
            for node_id in (edge.source_id, edge.target_id):
                if db.execute("SELECT 1 FROM cognitive_nodes WHERE id=?", (node_id,)).fetchone() is None:
                    raise KeyError(f"cognitive edge endpoint does not exist: {node_id}")
            db.execute(
                "INSERT INTO cognitive_edges VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source_id, target_id, relation) DO UPDATE SET evidence_ref=excluded.evidence_ref",
                (edge.source_id, edge.target_id, edge.relation.value, edge.evidence_ref),
            )

    def put_hypothesis(self, hypothesis: CognitiveEmotionalHypothesis) -> None:
        payload = json.dumps(_hypothesis_dict(hypothesis), sort_keys=True)
        with self._connect() as db:
            db.execute(
                "INSERT INTO cognitive_hypotheses VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET subject_id=excluded.subject_id, "
                "valid_from=excluded.valid_from, valid_until=excluded.valid_until, payload=excluded.payload",
                (hypothesis.id, hypothesis.subject_id, hypothesis.valid_from.isoformat(),
                 None if hypothesis.valid_until is None else hypothesis.valid_until.isoformat(), payload),
            )
            db.execute(
                "INSERT INTO cognitive_nodes VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, subject_id=excluded.subject_id, payload=excluded.payload",
                (hypothesis.id, CognitiveNodeKind.COGNITIVE_EMOTIONAL_HYPOTHESIS.value,
                 hypothesis.subject_id, payload),
            )

    def get_hypotheses(self, subject_id: str, *, valid_at: datetime | None = None) -> list[CognitiveEmotionalHypothesis]:
        _identity(subject_id, "subject_id")
        if valid_at is not None:
            _aware(valid_at, "valid_at")
            query = ("SELECT payload FROM cognitive_hypotheses WHERE subject_id=? "
                     "AND valid_from<=? AND (valid_until IS NULL OR valid_until>=?) ORDER BY id")
            params = (subject_id, valid_at.isoformat(), valid_at.isoformat())
        else:
            query, params = ("SELECT payload FROM cognitive_hypotheses WHERE subject_id=? ORDER BY id", (subject_id,))
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [_hypothesis_from(json.loads(row[0])) for row in rows]

    def put_snapshot(self, snapshot: BehaviorSnapshot) -> None:
        payload = json.dumps(asdict(snapshot), default=_json_default, sort_keys=True)
        with self._connect() as db:
            db.execute(
                "INSERT INTO behavior_snapshots VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET subject_id=excluded.subject_id, payload=excluded.payload",
                (snapshot.id, snapshot.subject_id, payload),
            )
            db.execute(
                "INSERT INTO cognitive_nodes VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, subject_id=excluded.subject_id, payload=excluded.payload",
                (snapshot.id, CognitiveNodeKind.SNAPSHOT.value, snapshot.subject_id, payload),
            )

    def get_snapshot(self, snapshot_id: str) -> BehaviorSnapshot | None:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM behavior_snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if row is None:
            return None
        raw = json.loads(row[0])
        return BehaviorSnapshot(raw["id"], raw["subject_id"],
            datetime.fromisoformat(raw["created_at"].replace("Z", "+00:00")),
            tuple(raw["behavior_ids"]), tuple(raw["hypothesis_ids"]), tuple(raw["baseline_refs"]))

    def get_baselines(self, subject_id: str) -> list[CognitiveNode]:
        _identity(subject_id, "subject_id")
        kinds = (CognitiveNodeKind.PROSODY_BASELINE.value, CognitiveNodeKind.TEXT_BASELINE.value)
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, kind, subject_id, payload FROM cognitive_nodes "
                "WHERE subject_id=? AND kind IN (?, ?) ORDER BY kind, id",
                (subject_id, *kinds),
            ).fetchall()
        return [CognitiveNode(row[0], CognitiveNodeKind(row[1]), row[2], json.loads(row[3])) for row in rows]

    def get_hypothesis_evidence(self, hypothesis_id: str) -> tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM cognitive_hypotheses WHERE id=?", (hypothesis_id,)).fetchone()
        if row is None:
            raise KeyError(hypothesis_id)
        raw = json.loads(row[0])
        item = _hypothesis_from(raw)
        return item.supporting_evidence, item.counter_evidence
