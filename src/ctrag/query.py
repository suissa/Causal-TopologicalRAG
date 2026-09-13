from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import QueryMode, RetrievalHit


class EvidenceLevel(str, Enum):
    OBSERVATIONAL_SUPPORT = "observational_support"
    INTERVENTIONAL_EVIDENCE = "interventional_evidence"
    COUNTERFACTUAL_GROUND_TRUTH = "counterfactual_ground_truth"


class CausalClaimError(ValueError):
    """Raised when output language exceeds the evidence contract."""


def validate_causal_claim(claim: str, evidence_level: EvidenceLevel) -> None:
    lowered = claim.casefold()
    causal_phrases = ("identified causal effect", "would have caused", "proved causality")
    if evidence_level is EvidenceLevel.OBSERVATIONAL_SUPPORT and any(
        phrase in lowered for phrase in causal_phrases
    ):
        raise CausalClaimError(
            "ordinary event logs provide observational_support only; "
            "interventional/counterfactual language requires labelled ground truth"
        )


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
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVATIONAL_SUPPORT

    def validate_claim(self, claim: str) -> None:
        validate_causal_claim(claim, self.evidence_level)

    @property
    def selected_anchors(self) -> tuple[str, ...]:
        return self.anchor_ids
