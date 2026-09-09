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
)


def test_reinforced_path_can_rerank_equal_causal_candidates() -> None:
    topology = CausalTopology()
    timestamp = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)
    topology.add_node(MemoryNode(id="root", text="root", timestamp=timestamp))
    topology.add_node(MemoryNode(id="a", text="candidate", timestamp=timestamp))
    topology.add_node(MemoryNode(id="b", text="candidate", timestamp=timestamp))
    edge_a = Edge("root", "a", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)
    edge_b = Edge("root", "b", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)
    topology.add_edge(edge_a)
    topology.add_edge(edge_b)

    base = CTRetriever(topology)
    before = base.search(
        "candidate",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["root"],
        exhaustive=True,
        k=2,
    )
    assert [hit.node.id for hit in before] == ["a", "b"]

    terrain = DynamicTerrain(topology)
    terrain.observe_transition("root", "b")
    terrain.observe_transition("root", "b")
    terrain_aware = TerrainAwareRetriever(base, terrain)
    after = terrain_aware.search(
        "candidate",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["root"],
        exhaustive=True,
        k=2,
    )

    assert [hit.node.id for hit in after] == ["b", "a"]
    assert after[0].components["terrain_influence"] > after[1].components["terrain_influence"]
    assert topology.outgoing("root", {EdgeKind.CAUSAL}) == [edge_a, edge_b]
