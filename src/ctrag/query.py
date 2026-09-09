from __future__ import annotations

from dataclasses import dataclass

from .models import QueryMode, RetrievalHit


@dataclass(slots=True, frozen=True)
class RetrievalStage:
    """One observable stage of the CT-RAG retrieval pipeline."""

    name: str
    node_ids: tuple[str, ...]
    description: str


@dataclass(slots=True)
class StagedRetrievalResult:
    """Explainable result of query-intent-aware staged retrieval."""

    query: str
    mode: QueryMode
    anchor_ids: tuple[str, ...]
    direction: str
    stages: tuple[RetrievalStage, ...]
    hits: list[RetrievalHit]
    observational_note: str | None = None

    @property
    def selected_anchors(self) -> tuple[str, ...]:
        return self.anchor_ids
