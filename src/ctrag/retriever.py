from __future__ import annotations

import math

from .adapters import EmbeddingProvider, IdfOverlapRetriever, LexicalRetriever, reciprocal_rank_fusion
from .embedding import HashingEmbedder, cosine_similarity
from .models import CausalPath, EdgeKind, QueryMode, RetrievalHit, RetrievalWeights
from .query import RetrievalStage, StagedRetrievalResult
from .topology import CausalTopology


class CTRetriever:
    def __init__(
        self,
        topology: CausalTopology,
        *,
        embedder: EmbeddingProvider | None = None,
        lexical_retriever: LexicalRetriever | None = None,
        hop_decay: float = 0.7,
    ) -> None:
        self.topology = topology
        self.embedder: EmbeddingProvider = embedder or HashingEmbedder()
        self.lexical_retriever: LexicalRetriever = lexical_retriever or IdfOverlapRetriever()
        self.hop_decay = hop_decay
        self._ensure_embeddings()

    def _ensure_embeddings(self) -> None:
        for node in self.topology.nodes.values():
            if node.embedding is None:
                node.embedding = self.embedder.embed(node.text)

    def _semantic_scores(self, query: str) -> dict[str, float]:
        query_embedding = self.embedder.embed(query)
        self._ensure_embeddings()
        return {
            node.id: max(0.0, cosine_similarity(query_embedding, node.embedding or ()))
            for node in self.topology.nodes.values()
        }

    def _lexical_scores(self, query: str) -> dict[str, float]:
        documents = {node.id: node.text for node in self.topology.nodes.values()}
        scores = self.lexical_retriever.score(query, documents)
        missing = set(documents).difference(scores)
        if missing:
            raise ValueError(f"lexical retriever omitted document ids: {sorted(missing)}")
        return {node_id: float(scores[node_id]) for node_id in documents}

    @staticmethod
    def _rank_scores(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]

    def rank_dense(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        return self._rank_scores(self._semantic_scores(query), k)

    def rank_lexical(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        return self._rank_scores(self._lexical_scores(query), k)

    def rank_hybrid_rrf(
        self,
        query: str,
        *,
        k: int = 10,
        rank_constant: int = 60,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        dense = [node_id for node_id, _ in self.rank_dense(query, k=len(self.topology.nodes))]
        lexical = [node_id for node_id, _ in self.rank_lexical(query, k=len(self.topology.nodes))]
        return reciprocal_rank_fusion([dense, lexical], rank_constant=rank_constant)[:k]

    def _direction_for(self, mode: QueryMode) -> str:
        if mode is QueryMode.WHY:
            return "in"
        if mode is QueryMode.WHAT_NEXT:
            return "out"
        return "both"

    def _staged_direction_for(self, mode: QueryMode) -> str:
        if mode in {QueryMode.WHY, QueryMode.COUNTERFACTUAL}:
            return "in"
        if mode in {QueryMode.WHAT_NEXT, QueryMode.RECOVERY}:
            return "out"
        return "both"

    @staticmethod
    def _causal_budget(
        direction: str,
        *,
        max_hops: int,
        ancestor_hops: int | None,
        descendant_hops: int | None,
    ) -> int:
        if direction == "in":
            return max_hops if ancestor_hops is None else ancestor_hops
        if direction == "out":
            return max_hops if descendant_hops is None else descendant_hops
        raise ValueError("causal budget direction must be in or out")

    def _temporal_scores(self) -> dict[str, float]:
        if not self.topology.nodes:
            return {}
        timestamps = [node.timestamp for node in self.topology.nodes.values()]
        newest = max(timestamps)
        oldest = min(timestamps)
        span = max(1.0, (newest - oldest).total_seconds())
        return {
            node.id: math.exp(-max(0.0, (newest - node.timestamp).total_seconds()) / span)
            for node in self.topology.nodes.values()
        }

    def _behavioral_affinity(self, candidate_id: str, anchor_ids: list[str]) -> float:
        candidate = self.topology.nodes[candidate_id]
        keys = ("execution_id", "intent_id", "actor_id", "action_id")
        best = 0.0
        for anchor_id in anchor_ids:
            anchor = self.topology.nodes[anchor_id]
            matches = 0
            available = 0
            for key in keys:
                left = candidate.metadata.get(key)
                right = anchor.metadata.get(key)
                if left is None or right is None:
                    continue
                available += 1
                if left == right:
                    matches += 1
            if available:
                best = max(best, matches / available)
        return best

    def _shared_basin_score(self, candidate_id: str, anchor_ids: list[str]) -> float:
        best = 0.0
        for anchor_id in anchor_ids:
            affinity = self.topology.shared_basin_affinity(candidate_id, anchor_id)
            best = max(best, affinity.score)
        return best

    def _topological_score(
        self,
        candidate_id: str,
        anchor_ids: list[str],
        *,
        direction: str,
        max_hops: int,
    ) -> tuple[float, str | None]:
        best_score = 0.0
        best_anchor: str | None = None
        kinds = {EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL, EdgeKind.TEMPORAL}
        for anchor_id in anchor_ids:
            distances = self.topology.distances(
                anchor_id,
                direction=direction,
                kinds=kinds,
                max_hops=max_hops,
            )
            hops = distances.get(candidate_id)
            if hops is None or hops == 0:
                continue
            score = math.exp(-self.hop_decay * hops)
            if score > best_score:
                best_score = score
                best_anchor = anchor_id
        return max(best_score, self._shared_basin_score(candidate_id, anchor_ids)), best_anchor

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
        directions = ["in"] if mode is QueryMode.WHY else ["out"] if mode is QueryMode.WHAT_NEXT else ["in", "out"]
        best_score = 0.0
        best_hops: int | None = None
        best_anchor: str | None = None
        best_path: CausalPath | None = None
        for anchor_id in anchor_ids:
            for direction in directions:
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

    def _select_anchor_ids(
        self,
        query: str,
        *,
        anchor_k: int,
        anchor_ids: list[str] | None,
    ) -> list[str]:
        if anchor_ids is not None:
            missing = [node_id for node_id in anchor_ids if node_id not in self.topology.nodes]
            if missing:
                raise KeyError(f"unknown anchor ids: {missing}")
            return list(anchor_ids)
        semantic = self._semantic_scores(query)
        lexical = self._lexical_scores(query)
        ranked = sorted(
            self.topology.nodes,
            key=lambda node_id: (-(0.65 * semantic[node_id] + 0.35 * lexical[node_id]), node_id),
        )
        return ranked[: max(1, anchor_k)]

    def _recovery_prior(self, candidate_id: str, anchor_ids: list[str], max_hops: int) -> float:
        node = self.topology.nodes[candidate_id]
        status = str(node.metadata.get("status", "")).lower()
        event_type = str(node.metadata.get("event_type", "")).lower()
        successful = status in {"ok", "success", "succeeded", "completed", "recovered", "healed"}
        successful = successful or any(token in event_type for token in ("healed", "recovered", "completed", "succeeded"))
        if not successful:
            return 0.0
        best = 0.0
        for anchor_id in anchor_ids:
            path = self.topology.causal_path_evidence(anchor_id, candidate_id, direction="out", max_hops=max_hops)
            if path is not None and path.hops > 0:
                best = max(best, path.aggregate_confidence * math.exp(-self.hop_decay * (path.hops - 1)))
        return best

    def _counterfactual_prior(self, candidate_id: str, anchor_ids: list[str], max_hops: int) -> float:
        outgoing = self.topology.outgoing(candidate_id, {EdgeKind.CAUSAL})
        if len({edge.target for edge in outgoing}) < 2:
            return 0.0
        best = 0.0
        for anchor_id in anchor_ids:
            path = self.topology.causal_path_evidence(anchor_id, candidate_id, direction="in", max_hops=max_hops)
            if path is not None and path.hops > 0:
                best = max(best, path.aggregate_confidence * math.exp(-self.hop_decay * (path.hops - 1)))
        return best

    def search_staged(
        self,
        query: str,
        *,
        mode: QueryMode,
        k: int = 5,
        anchor_k: int = 3,
        anchor_ids: list[str] | None = None,
        max_hops: int = 4,
        ancestor_hops: int | None = None,
        descendant_hops: int | None = None,
        weights: RetrievalWeights | None = None,
    ) -> StagedRetrievalResult:
        """Run query-intent-aware retrieval while exposing each navigation stage.

        `search()` remains the historical baseline used by the published benchmark.
        This method layers explicit stage telemetry and mode-specific reranking on
        top without changing those baseline results.
        """
        if not self.topology.nodes or k <= 0:
            return StagedRetrievalResult(query, mode, (), self._staged_direction_for(mode), (), [])

        selected = self._select_anchor_ids(query, anchor_k=anchor_k, anchor_ids=anchor_ids)
        direction = self._staged_direction_for(mode)
        stage_kinds = {EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL, EdgeKind.TEMPORAL}

        topology_expansion: set[str] = set()
        basin_expansion: set[str] = set()
        causal_expansion: set[str] = set()
        for anchor_id in selected:
            topology_expansion.update(self.topology.neighborhood(
                anchor_id,
                direction=direction,
                kinds=stage_kinds,
                max_hops=max_hops,
            ))
            for attractor_id in self.topology.basin_memberships(anchor_id, max_hops=max_hops):
                basin_expansion.update(self.topology.basin(attractor_id, max_hops=max_hops))
            causal_expansion.update(self.topology.neighborhood(
                anchor_id,
                direction=direction,
                kinds={EdgeKind.CAUSAL},
                max_hops=max_hops,
            ))

        base_hits = self.search(
            query,
            mode=mode,
            k=max(k * 4, k),
            anchor_k=anchor_k,
            anchor_ids=selected,
            max_hops=max_hops,
            ancestor_hops=ancestor_hops,
            descendant_hops=descendant_hops,
            weights=weights,
            exhaustive=True,
        )

        reranked: list[RetrievalHit] = []
        mode_nodes: list[str] = []
        for hit in base_hits:
            prior = 0.0
            if mode is QueryMode.RECOVERY:
                prior = self._recovery_prior(hit.node.id, selected, max_hops)
            elif mode is QueryMode.COUNTERFACTUAL:
                prior = self._counterfactual_prior(hit.node.id, selected, max_hops)
            components = dict(hit.components)
            components["mode_prior"] = prior
            if prior > 0:
                mode_nodes.append(hit.node.id)
            reranked.append(RetrievalHit(
                node=hit.node,
                score=hit.score + 0.25 * prior,
                components=components,
                anchor_id=hit.anchor_id,
                causal_hops=hit.causal_hops,
                causal_path=hit.causal_path,
            ))

        reranked.sort(key=lambda hit: (-hit.score, hit.node.id))
        stages = (
            RetrievalStage("anchor_search", tuple(selected), "Global semantic/lexical anchor selection."),
            RetrievalStage(
                "basin_topology_expansion",
                tuple(sorted(topology_expansion | basin_expansion)),
                "Local topology and any registered basin memberships around selected anchors.",
            ),
            RetrievalStage(
                "causal_traversal",
                tuple(sorted(causal_expansion)),
                f"Directed causal traversal ({direction}) for query mode {mode.value}.",
            ),
            RetrievalStage(
                "mode_rerank",
                tuple(sorted(set(mode_nodes))),
                "Mode-specific prior: successful future recovery paths or historical divergence points.",
            ),
        )
        note = None
        if mode is QueryMode.COUNTERFACTUAL:
            note = (
                "Counterfactual results are observational/hypothesis support only. "
                "Historical divergence does not identify intervention effects or prove causality."
            )
        return StagedRetrievalResult(
            query=query,
            mode=mode,
            anchor_ids=tuple(selected),
            direction=direction,
            stages=stages,
            hits=reranked[:k],
            observational_note=note,
        )

    def why(self, anchor_id: str, query: str = "why did this happen?", *, k: int = 5) -> StagedRetrievalResult:
        return self.search_staged(query, mode=QueryMode.WHY, anchor_ids=[anchor_id], k=k)

    def what_next(self, anchor_id: str, query: str = "what happened next?", *, k: int = 5) -> StagedRetrievalResult:
        return self.search_staged(query, mode=QueryMode.WHAT_NEXT, anchor_ids=[anchor_id], k=k)

    def recovery(self, anchor_id: str, query: str = "how was this recovered?", *, k: int = 5) -> StagedRetrievalResult:
        return self.search_staged(query, mode=QueryMode.RECOVERY, anchor_ids=[anchor_id], k=k)

    def counterfactual(self, anchor_id: str, query: str = "where did comparable trajectories diverge?", *, k: int = 5) -> StagedRetrievalResult:
        return self.search_staged(query, mode=QueryMode.COUNTERFACTUAL, anchor_ids=[anchor_id], k=k)

    def search(
        self,
        query: str,
        *,
        mode: QueryMode = QueryMode.SIMILAR,
        k: int = 5,
        anchor_k: int = 3,
        anchor_ids: list[str] | None = None,
        max_hops: int = 4,
        ancestor_hops: int | None = None,
        descendant_hops: int | None = None,
        weights: RetrievalWeights | None = None,
        include_anchors: bool = False,
        exhaustive: bool = False,
    ) -> list[RetrievalHit]:
        if k <= 0:
            return []
        if not self.topology.nodes:
            return []
        for name, value in (("max_hops", max_hops), ("ancestor_hops", ancestor_hops), ("descendant_hops", descendant_hops)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        semantic = self._semantic_scores(query)
        lexical = self._lexical_scores(query)
        temporal = self._temporal_scores()
        weights = weights or RetrievalWeights.for_mode(mode)

        if not math.isclose(weights.total(), 1.0, abs_tol=1e-9):
            raise ValueError("retrieval weights must sum to 1")

        if anchor_ids is None:
            ranked_anchors = sorted(
                self.topology.nodes,
                key=lambda node_id: 0.65 * semantic[node_id] + 0.35 * lexical[node_id],
                reverse=True,
            )
            anchor_ids = ranked_anchors[: max(1, anchor_k)]
        else:
            missing = [node_id for node_id in anchor_ids if node_id not in self.topology.nodes]
            if missing:
                raise KeyError(f"unknown anchor ids: {missing}")

        direction = self._direction_for(mode)
        global_ranked = sorted(
            self.topology.nodes,
            key=lambda node_id: 0.65 * semantic[node_id] + 0.35 * lexical[node_id],
            reverse=True,
        )
        candidate_ids = set(global_ranked[: max(20, k * 4)])
        expansion_hops = max(
            max_hops,
            ancestor_hops if ancestor_hops is not None else max_hops,
            descendant_hops if descendant_hops is not None else max_hops,
        )
        for anchor_id in anchor_ids:
            candidate_ids.update(
                self.topology.neighborhood(
                    anchor_id,
                    direction=direction,
                    kinds={EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL, EdgeKind.TEMPORAL},
                    max_hops=expansion_hops,
                )
            )

        if exhaustive:
            candidate_ids = set(self.topology.nodes)

        hits: list[RetrievalHit] = []
        for candidate_id in candidate_ids:
            if not include_anchors and candidate_id in anchor_ids:
                continue
            topological, topo_anchor = self._topological_score(
                candidate_id,
                anchor_ids,
                direction=direction,
                max_hops=expansion_hops,
            )
            causal, causal_hops, causal_anchor, causal_path = self._causal_score(
                candidate_id,
                anchor_ids,
                mode=mode,
                max_hops=max_hops,
                ancestor_hops=ancestor_hops,
                descendant_hops=descendant_hops,
            )
            behavioral = self._behavioral_affinity(candidate_id, anchor_ids)
            components = {
                "semantic": semantic[candidate_id],
                "lexical": lexical[candidate_id],
                "causal": causal,
                "topological": topological,
                "temporal": temporal[candidate_id],
                "behavioral": behavioral,
            }
            score = (
                weights.semantic * components["semantic"]
                + weights.lexical * components["lexical"]
                + weights.causal * components["causal"]
                + weights.topological * components["topological"]
                + weights.temporal * components["temporal"]
                + weights.behavioral * components["behavioral"]
            )
            hits.append(
                RetrievalHit(
                    node=self.topology.nodes[candidate_id],
                    score=score,
                    components=components,
                    anchor_id=causal_anchor or topo_anchor,
                    causal_hops=causal_hops,
                    causal_path=causal_path,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.node.id))
        return hits[:k]
