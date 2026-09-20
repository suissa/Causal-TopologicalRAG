from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EvidenceShape(str, Enum):
    NARRATIVE = "narrative"
    TABLE = "table"
    CONFIG_TREE = "config_tree"
    TRACE = "trace"
    METRIC_SERIES = "metric_series"
    EVENT = "event"
    LOG = "log"
    CODE = "code"
    STRUCTURED_PAYLOAD = "structured_payload"


class LocalRelationKind(str, Enum):
    """Artifact-local relations. This enum intentionally has no CAUSAL member."""

    CONFIG_PARENT = "config_parent"
    TRACE_PARENT = "trace_parent"
    SERIES_ADJACENT = "series_adjacent"


@dataclass(slots=True, frozen=True)
class RawEvidence:
    source_id: str
    content: Any
    source_uri: str | None = None
    filename: str | None = None
    media_type: str | None = None
    observed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")


@dataclass(slots=True, frozen=True)
class EvidenceRoute:
    shape: EvidenceShape | None
    confidence: float
    method_id: str
    adapter_id: str | None
    preserve: tuple[str, ...]
    source_metadata: dict[str, Any] = field(default_factory=dict, hash=False)
    fallback_required: bool = False
    classifier_version: str = "deterministic-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.method_id.strip():
            raise ValueError("method_id must be non-empty")
        if len(self.preserve) != len(set(self.preserve)):
            raise ValueError("preserve fields must be unique")

    @property
    def is_determined(self) -> bool:
        return self.shape is not None


@dataclass(slots=True, frozen=True)
class LocalRelation:
    kind: LocalRelationKind
    source_anchor_id: str
    target_anchor_id: str
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)


@dataclass(slots=True, frozen=True)
class TypedEvidenceAnchor:
    id: str
    shape: EvidenceShape
    source_id: str
    content_summary: str
    structural_locator: dict[str, Any]
    preserved_fields: dict[str, Any]
    retrieval_features: dict[str, Any]
    provenance: dict[str, Any]
    confidence: float
    classifier_version: str
    adapter_version: str
    source_uri: str | None = None
    observed_at: datetime | None = None
    local_relations: tuple[LocalRelation, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("anchor id must be non-empty")
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
