from __future__ import annotations

from dataclasses import replace

from .models import QueryMode, RetrievalHit
from .query import StagedRetrievalResult
from .retriever import CTRetriever
from .terrain import DynamicTerrain


class TerrainAwareRetriever:
    """Apply empirical terrain influence as a non-authoritative final reranker.

    The wrapped CTRetriever remains unchanged. Terrain influence is exposed as a
    separate score component so historical baseline results stay reproducible.
    """

    def __init__(self, retriever: CTRetriever, terrain: DynamicTerrain) -> None:
        if retriever.topology is not terrain.topology:
            raise ValueError("retriever and terrain must reference the same topology")
        self.retriever = retriever
        self.terrain = terrain

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

    def search(self, query: str, *, mode: QueryMode = QueryMode.SIMILAR, k: int = 5, **kwargs) -> list[RetrievalHit]:
        hits = self.retriever.search(query, mode=mode, k=max(k * 4, k), **kwargs)
        return self._rerank(hits, k)

    def search_staged(self, query: str, *, mode: QueryMode, k: int = 5, **kwargs) -> StagedRetrievalResult:
        result = self.retriever.search_staged(query, mode=mode, k=max(k * 4, k), **kwargs)
        result.hits = self._rerank(result.hits, k)
        return result
