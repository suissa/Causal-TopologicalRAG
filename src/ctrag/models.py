from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _validate_identity(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _validate_timestamp(value: datetime) -> None:
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")


class EdgeKind(str, Enum):
    SEMANTIC = "semantic"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"


class CausalProvenance(str, Enum):
    EXECUTION = "execution"
    DEPENDENCY = "dependency"
    WORKFLOW = "workflow"
    EVENT = "event"
    INFERRED = "inferred"
    HYPOTHESIZED = "hypothesized"


class QueryMode(str, Enum):
    SIMILAR = "similar"
    WHY = "why"
    WHAT_NEXT = "what_next"
    RECOVERY = "recovery"
    COUNTERFACTUAL = "counterfactual"


@dataclass(slots=True, frozen=True)
class EdgeEvidence:
    id: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _validate_identity(self.id, "evidence id")
        if self.source is not None:
            _validate_identity(self.source, "evidence source")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EdgeEvidence":
        return cls(
            id=str(raw["id"]),
            source=None if raw.get("source") is None else str(raw["source"]),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(slots=True)
class MemoryNode:
    id: str
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    is_attractor: bool = False

    def __post_init__(self) -> None:
        _validate_identity(self.id, "node id")
        if not isinstance(self.text, str):
            raise TypeError("node text must be a string")
        _validate_timestamp(self.timestamp)
        if self.embedding is not None and any(not math.isfinite(value) for value in self.embedding):
            raise ValueError("embedding values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "is_attractor": self.is_attractor,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MemoryNode":
        timestamp = raw.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timestamp is None:
            raise ValueError("serialized node requires timestamp")
        embedding = raw.get("embedding")
        return cls(
            id=str(raw["id"]),
            text=str(raw.get("text", "")),
            timestamp=timestamp,
            metadata=dict(raw.get("metadata") or {}),
            embedding=None if embedding is None else tuple(float(value) for value in embedding),
            is_attractor=bool(raw.get("is_attractor", False)),
        )


@dataclass(slots=True, frozen=True)
class Edge:
    source: str
    target: str
    kind: EdgeKind
    weight: float = 1.0
    provenance: CausalProvenance | None = None
    confidence: float = 1.0
    evidence: tuple[EdgeEvidence, ...] = ()
    provenance_metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _validate_identity(self.source, "edge source")
        _validate_identity(self.target, "edge target")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("edge confidence must be finite and between 0 and 1")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("edge weight must be finite and non-negative")
        if self.kind is EdgeKind.CAUSAL and self.provenance is None:
            raise ValueError("causal edges require provenance")
        if self.kind is not EdgeKind.CAUSAL and self.provenance is not None:
            raise ValueError("causal provenance is only valid for causal edges")
        if self.kind is not EdgeKind.CAUSAL and (self.evidence or self.provenance_metadata):
            raise ValueError("causal evidence/provenance metadata is only valid for causal edges")
        evidence_ids = [item.id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("edge evidence ids must be unique")

    def identity(self) -> tuple[str, str, EdgeKind, CausalProvenance | None]:
        return self.source, self.target, self.kind, self.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "weight": self.weight,
            "provenance": None if self.provenance is None else self.provenance.value,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "provenance_metadata": dict(self.provenance_metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Edge":
        provenance = raw.get("provenance")
        return cls(
            source=str(raw["source"]),
            target=str(raw["target"]),
            kind=EdgeKind(raw["kind"]),
            weight=float(raw.get("weight", 1.0)),
            provenance=None if provenance is None else CausalProvenance(provenance),
            confidence=float(raw.get("confidence", 1.0)),
            evidence=tuple(EdgeEvidence.from_dict(item) for item in raw.get("evidence", [])),
            provenance_metadata=dict(raw.get("provenance_metadata") or {}),
        )


@dataclass(slots=True, frozen=True)
class CausalPath:
    """Explainable causal path selected for one anchor/candidate pair."""

    anchor_id: str
    candidate_id: str
    direction: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    best_confidence: float
    aggregate_confidence: float

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def evidence(self) -> tuple[EdgeEvidence, ...]:
        seen: set[str] = set()
        result: list[EdgeEvidence] = []
        for edge in self.edges:
            for item in edge.evidence:
                if item.id not in seen:
                    seen.add(item.id)
                    result.append(item)
        return tuple(result)

    @property
    def provenances(self) -> tuple[CausalProvenance, ...]:
        return tuple(edge.provenance for edge in self.edges if edge.provenance is not None)


@dataclass(slots=True, frozen=True)
class RetrievalWeights:
    semantic: float
    lexical: float
    causal: float
    topological: float
    temporal: float
    behavioral: float

    def __post_init__(self) -> None:
        for name in ("semantic", "lexical", "causal", "topological", "temporal", "behavioral"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"retrieval weight {name} must be finite and non-negative")

    @classmethod
    def for_mode(cls, mode: QueryMode) -> "RetrievalWeights":
        presets = {
            QueryMode.SIMILAR: cls(0.50, 0.30, 0.05, 0.10, 0.05, 0.00),
            QueryMode.WHY: cls(0.15, 0.10, 0.35, 0.20, 0.05, 0.15),
            QueryMode.WHAT_NEXT: cls(0.15, 0.10, 0.30, 0.25, 0.05, 0.15),
            QueryMode.RECOVERY: cls(0.15, 0.10, 0.25, 0.15, 0.05, 0.30),
            QueryMode.COUNTERFACTUAL: cls(0.20, 0.10, 0.25, 0.15, 0.05, 0.25),
        }
        return presets[mode]

    def total(self) -> float:
        return (
            self.semantic
            + self.lexical
            + self.causal
            + self.topological
            + self.temporal
            + self.behavioral
        )


@dataclass(slots=True)
class RetrievalHit:
    node: MemoryNode
    score: float
    components: dict[str, float]
    anchor_id: str | None = None
    causal_hops: int | None = None
    causal_path: CausalPath | None = None
