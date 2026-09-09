from __future__ import annotations

import math

from .models import CausalPath, QueryMode
from .retriever import CTRetriever as _BaseCTRetriever


class CTRetriever(_BaseCTRetriever):
    """CTRetriever with one explicit direction policy shared by all query stages.

    This class is exported as the public CTRetriever by :mod:`ctrag`.  It exists
    as a small compatibility layer so the scientific audit can correct the
    historical v0.1 bidirectional RECOVERY/COUNTERFACTUAL behavior without
    rewriting unrelated retrieval logic.
    """

    @staticmethod
    def _query_directions(mode: QueryMode) -> tuple[str, ...]:
        if mode in {QueryMode.WHY, QueryMode.COUNTERFACTUAL}:
            return ("in",)
        if mode in {QueryMode.WHAT_NEXT, QueryMode.RECOVERY}:
            return ("out",)
        return ("in", "out")

    def _direction_for(self, mode: QueryMode) -> str:
        directions = self._query_directions(mode)
        return directions[0] if len(directions) == 1 else "both"

    def _causal_score(
        self,
        candidate_id: str,
        anchor_ids: list[str],
        *,
        mode: QueryMode,
        max_hops: int,
        ancestor_hops: int | None,
        descendant_hops: int | None,
    ) -> tuple[float, int | None, str | None, CausalPath | None]:
        best_score = 0.0
        best_hops: int | None = None
        best_anchor: str | None = None
        best_path: CausalPath | None = None
        for anchor_id in anchor_ids:
            for direction in self._query_directions(mode):
                budget = self._causal_budget(
                    direction,
                    max_hops=max_hops,
                    ancestor_hops=ancestor_hops,
                    descendant_hops=descendant_hops,
                )
                path = self.topology.causal_path_evidence(
                    anchor_id,
                    candidate_id,
                    direction=direction,
                    max_hops=budget,
                )
                if path is None or path.hops == 0:
                    continue
                score = path.aggregate_confidence * math.exp(-self.hop_decay * (path.hops - 1))
                if score > best_score:
                    best_score = score
                    best_hops = path.hops
                    best_anchor = anchor_id
                    best_path = path
        return best_score, best_hops, best_anchor, best_path
