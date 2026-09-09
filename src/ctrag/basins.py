from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class AttractorDescriptor:
    """First-class metadata for a manually declared or empirically found attractor."""

    node_id: str
    confidence: float = 1.0
    origin: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("attractor node_id must be non-empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attractor confidence must be finite and between 0 and 1")
        if not self.origin.strip():
            raise ValueError("attractor origin must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "confidence": self.confidence,
            "origin": self.origin,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AttractorDescriptor":
        return cls(
            node_id=str(raw["node_id"]),
            confidence=float(raw.get("confidence", 1.0)),
            origin=str(raw.get("origin", "manual")),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(slots=True, frozen=True)
class BasinAffinity:
    """Explainable shared-basin evidence between two nodes."""

    left_id: str
    right_id: str
    shared_attractors: tuple[str, ...]
    score: float
