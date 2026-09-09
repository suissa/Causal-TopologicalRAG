from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


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


@dataclass(slots=True)
class MemoryNode:
    id: str
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    is_attractor: bool = False


@dataclass(slots=True, frozen=True)
class Edge:
    source: str
    target: str
    kind: EdgeKind
    weight: float = 1.0
    provenance: CausalProvenance | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("edge confidence must be between 0 and 1")
        if self.weight < 0:
            raise ValueError("edge weight cannot be negative")
        if self.kind is EdgeKind.CAUSAL and self.provenance is None:
            raise ValueError("causal edges require provenance")
        if self.kind is not EdgeKind.CAUSAL and self.provenance is not None:
            raise ValueError("causal provenance is only valid for causal edges")


@dataclass(slots=True, frozen=True)
class RetrievalWeights:
    semantic: float
    lexical: float
    causal: float
    topological: float
    temporal: float
    behavioral: float

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
