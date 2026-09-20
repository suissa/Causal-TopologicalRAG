from __future__ import annotations

import math
from dataclasses import replace

from .models import EdgeKind, QueryMode, RetrievalHit
from .query import StagedRetrievalResult
from .retriever import CTRetriever
from .terrain import DynamicTerrain


_SUCCESS_STATUSES = {"ok", "success", "succeeded", "completed", "recovered", "healed"}


class TerrainAwareRetriever:
    """Apply empirical terrain influence as a non-authoritative reranking signal.

    For ordinary modes, terrain influence remains a multiplicative navigational
    prior. RECOVERY is intentionally different: low current influence must not
    veto preserved historical recovery evidence. Recovery ranking therefore uses
    empirical branch success and observed support, while terrain is exposed as a
    component rather than a hard multiplier.
    """

    def __init__(self, retriever: CTRetriever, terrain: DynamicTerrain) -> None:
        if retriever.topology is not terrain.topology:
            raise ValueError("retriever and terrain must reference the same topology")
        self.retriever = retriever
        self.terrain = terrain

    @staticmethod
    def _is_success_node(hit: RetrievalHit) -> bool:
        status = str(hit.node.metadata.get("status", "")).lower()
        event_type = str(hit.node.metadata.get("event_type", "")).lower()
        return status in _SUCCESS_STATUSES or any(
            token in event_type for token in ("healed", "recovered", "completed", "succeeded")
        )

    @staticmethod
    def _wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
        """95% Wilson lower bound for a binomial success probability."""
        if total <= 0:
            return 0.0
        p = successes / total
        z2 = z * z
        denominator = 1.0 + z2 / total
        centre = p + z2 / (2.0 * total)
        margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
        return max(0.0, (centre - margin) / denominator)

    def _branch_success_stats(self, hit: RetrievalHit) -> tuple[int, int]:
        """Count successful and total outcomes at the final causal branch.

        Only candidates causally reachable from the current anchor can receive
        recovery evidence. This is the topological conditioning that a global
        raw-success baseline lacks.

        The last edge source is treated as the branch point immediately preceding
        the terminal candidate. Counts come only from observed terrain history;
        unobserved structural edges contribute zero observations.

        Returns (successful_outcomes, total_observed_outcomes). A successful
        terminal with no causal path from the current anchor is ineligible.
        """
        path = hit.causal_path
        if path is None or not path.edges or not self._is_success_node(hit):
            return 0, 0

        branch_source = path.edges[-1].source
        outgoing = self.terrain.topology.outgoing(branch_source, {EdgeKind.CAUSAL})
        total = 0
        successful = 0
        for edge in outgoing:
            count = self.terrain.transition_count(edge)
            if count <= 0:
                continue
            total += count
            target = self.terrain.topology.nodes[edge.target]
            status = str(target.metadata.get("status", "")).lower()
            event_type = str(target.metadata.get("event_type", "")).lower()
            target_success = status in _SUCCESS_STATUSES or any(
                token in event_type for token in ("healed", "recovered", "completed", "succeeded")
            )
            if target_success:
                successful += count
        return successful, total

    def _rerank(self, hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        reranked: list[RetrievalHit] = []
        for hit in hits:
            influence = self.terrain.path_influence(hit.causal_path)
            components = dict(hit.components)
            components["terrain_influence"] = influence
            reranked.append(RetrievalHit(
                node=hit.node,
                score=hit.score * influence,
                components=components,
                anchor_id=hit.anchor_id,
                causal_hops=hit.causal_hops,
                causal_path=hit.causal_path,
            ))
        reranked.sort(key=lambda hit: (-hit.score, hit.node.id))
        return reranked[:k]

    def _rerank_recovery(self, hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        """Rank historical recovery evidence without letting erosion act as a veto.

        Ordering components:
        - base CT-RAG recovery score;
        - conditional historical branch success rate;
        - bounded observation support.

        Current terrain influence is retained for explanation only and never
        multiplied into the recovery score. Historical support is represented
        by the 95% Wilson lower confidence bound, not by an ad-hoc sample-size
        multiplier.
        """
        reranked: list[RetrievalHit] = []
        for hit in hits:
            influence = self.terrain.path_influence(hit.causal_path)
            successes, observations = self._branch_success_stats(hit)
            success_rate = 0.0 if observations <= 0 else successes / observations
            wilson_lower = self._wilson_lower_bound(successes, observations)
            reachable = hit.causal_path is not None and hit.causal_path.hops > 0
            components = dict(hit.components)
            components["terrain_influence"] = influence
            components["historical_success_rate"] = success_rate
            components["historical_support_n"] = float(observations)
            components["historical_wilson_lower_95"] = wilson_lower
            components["recovery_causally_reachable"] = 1.0 if reachable else 0.0

            # The recovery evidence score is conservative in small-N regimes.
            # Wilson lower bound folds rate and sample support into one quantity.
            empirical_recovery = wilson_lower if self._is_success_node(hit) and reachable else 0.0
            score = hit.score + 0.35 * empirical_recovery
            reranked.append(RetrievalHit(
                node=hit.node,
                score=score,
                components=components,
                anchor_id=hit.anchor_id,
                causal_hops=hit.causal_hops,
                causal_path=hit.causal_path,
            ))
        reranked.sort(key=lambda hit: (-hit.score, hit.node.id))
        return reranked[:k]

    def search(self, query: str, *, mode: QueryMode = QueryMode.SIMILAR, k: int = 5, **kwargs) -> list[RetrievalHit]:
        hits = self.retriever.search(query, mode=mode, k=max(k * 4, k), **kwargs)
        if mode is QueryMode.RECOVERY:
            return self._rerank_recovery(hits, k)
        return self._rerank(hits, k)

    def search_staged(self, query: str, *, mode: QueryMode, k: int = 5, **kwargs) -> StagedRetrievalResult:
        result = self.retriever.search_staged(query, mode=mode, k=max(k * 4, k), **kwargs)
        if mode is QueryMode.RECOVERY:
            result.hits = self._rerank_recovery(result.hits, k)
        else:
            result.hits = self._rerank(result.hits, k)
        return result
