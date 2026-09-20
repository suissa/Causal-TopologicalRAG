from __future__ import annotations

from datetime import datetime, timezone

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    DynamicTerrain,
    Edge,
    EdgeKind,
    MemoryNode,
    QueryMode,
    TerrainAwareRetriever,
    TerrainConfig,
)


def _edge(source: str, target: str) -> Edge:
    return Edge(source, target, EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)


def test_recovery_rerank_uses_historical_success_without_terrain_veto() -> None:
    topology = CausalTopology()
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    for node_id, status in (
        ("root", "retry"),
        ("provider", "fallback"),
        ("provider_ok", "recovered"),
        ("manual", "manual_patch"),
        ("manual_ok", "recovered"),
        ("manual_fail", "failed"),
    ):
        topology.add_node(MemoryNode(
            id=node_id,
            text="payment recovery",
            timestamp=now,
            metadata={"status": status},
        ))

    p1, p2 = _edge("root", "provider"), _edge("provider", "provider_ok")
    m1 = _edge("root", "manual")
    m_ok, m_fail = _edge("manual", "manual_ok"), _edge("manual", "manual_fail")
    for edge in (p1, p2, m1, m_ok, m_fail):
        topology.add_edge(edge)

    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(reinforcement_step=0.5, decay_rate=0.2),
    )
    for _ in range(8):
        terrain.reinforce(p1)
        terrain.reinforce(p2)
    for _ in range(1):
        terrain.reinforce(m1)
        terrain.reinforce(m_ok)
    for _ in range(9):
        terrain.reinforce(m1)
        terrain.reinforce(m_fail)

    # Erode both recovery branches so current terrain is not the deciding signal.
    terrain.decay(25.0)

    result = TerrainAwareRetriever(CTRetriever(topology), terrain).search_staged(
        "payment recovery",
        mode=QueryMode.RECOVERY,
        anchor_ids=["root"],
        k=8,
    )
    recovered = [hit for hit in result.hits if hit.node.metadata.get("status") == "recovered"]

    assert [hit.node.id for hit in recovered][:2] == ["provider_ok", "manual_ok"]
    provider = next(hit for hit in recovered if hit.node.id == "provider_ok")
    manual = next(hit for hit in recovered if hit.node.id == "manual_ok")
    assert provider.components["historical_success_rate"] == 1.0
    assert manual.components["historical_success_rate"] == 0.1
    assert provider.score > manual.score


def test_wilson_support_penalizes_one_for_one() -> None:
    assert TerrainAwareRetriever._wilson_lower_bound(8, 8) > TerrainAwareRetriever._wilson_lower_bound(1, 1)
    assert TerrainAwareRetriever._wilson_lower_bound(8, 10) > TerrainAwareRetriever._wilson_lower_bound(1, 10)
