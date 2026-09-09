from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

from .embedding import HashingEmbedder, cosine_similarity, tokenize
from .models import EdgeKind, QueryMode, RetrievalHit, RetrievalWeights
from .topology import CausalTopology


class CTRetriever:
    def __init__(
        self,
        topology: CausalTopology,
        *,
        embedder: HashingEmbedder | None = None,
        hop_decay: float = 0.7,
    ) -> None:
        self.topology = topology
        self.embedder = embedder or HashingEmbedder()
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
        query_terms = set(tokenize(query))
        if not query_terms:
            return {node_id: 0.0 for node_id in self.topology.nodes}

        documents = {node.id: set(tokenize(node.text)) for node in self.topology.nodes.values()}
        document_frequency = Counter(
            term for terms in documents.values() for term in query_terms.intersection(terms)
        )
        total = max(1, len(documents))
        idf = {
            term: math.log((total + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
            for term in query_terms
        }
        denominator = sum(idf.values()) or 1.0
        return {
            node_id: sum(idf[term] for term in query_terms.intersection(terms)) / denominator
            for node_id, terms in documents.items()
        }

    def _direction_for(self, mode: QueryMode) -> str:
        if mode is QueryMode.WHY:
            return "in"
        if mode is QueryMode.WHAT_NEXT:
            return "out"
        return "both"

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
        candidate_attractors = self.topology.attractors_for(candidate_id)
        if not candidate_attractors:
            return 0.0
        for anchor_id in anchor_ids:
            if candidate_attractors.intersection(self.topology.attractors_for(anchor_id)):
                return 0.6
        return 0.0

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
    ) -> tuple[float, int | None, str | None]:
        directions = ["in"] if mode is QueryMode.WHY else ["out"] if mode is QueryMode.WHAT_NEXT else ["in", "out"]
        best_score = 0.0
        best_hops: int | None = None
        best_anchor: str | None = None
        for anchor_id in anchor_ids:
            for direction in directions:
                hops, confidence = self.topology.causal_path_confidence(
                    anchor_id,
                    candidate_id,
                    direction=direction,
                    max_hops=max_hops,
                )
                if hops is None or hops == 0:
                    continue
                score = confidence * math.exp(-self.hop_decay * (hops - 1))
                if score > best_score:
                    best_score = score
                    best_hops = hops
                    best_anchor = anchor_id
        return best_score, best_hops, best_anchor

    def search(
        self,
        query: str,
        *,
        mode: QueryMode = QueryMode.SIMILAR,
        k: int = 5,
        anchor_k: int = 3,
        anchor_ids: list[str] | None = None,
        max_hops: int = 4,
        weights: RetrievalWeights | None = None,
        include_anchors: bool = False,
        exhaustive: bool = False,
    ) -> list[RetrievalHit]:
        if k <= 0:
            return []
        if not self.topology.nodes:
            return []

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
        for anchor_id in anchor_ids:
            candidate_ids.update(
                self.topology.neighborhood(
                    anchor_id,
                    direction=direction,
                    kinds={EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL, EdgeKind.TEMPORAL},
                    max_hops=max_hops,
                )
            )

        # Controlled ablations must not inherit semantic candidate pruning.
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
                max_hops=max_hops,
            )
            causal, causal_hops, causal_anchor = self._causal_score(
                candidate_id,
                anchor_ids,
                mode=mode,
                max_hops=max_hops,
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
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.node.id))
        return hits[:k]
