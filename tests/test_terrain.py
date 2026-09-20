from __future__ import annotations

from ctrag import (
    CausalProvenance,
    CausalTopology,
    DynamicTerrain,
    Edge,
    EdgeKind,
    MemoryNode,
    SurpriseSignal,
    SurpriseSource,
    TerrainConfig,
)


def causal(source: str, target: str) -> Edge:
    return Edge(
        source=source,
        target=target,
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    )


def simple_topology() -> tuple[CausalTopology, Edge, Edge]:
    topology = CausalTopology()
    for node_id in ("a", "b", "sink"):
        topology.add_node(MemoryNode(id=node_id, text=node_id))
    first = causal("a", "b")
    second = causal("b", "sink")
    topology.add_edge(first)
    topology.add_edge(second)
    return topology, first, second


def test_repeated_trajectories_reinforce_navigation_without_rewriting_edges() -> None:
    topology, first, second = simple_topology()
    terrain = DynamicTerrain(topology, config=TerrainConfig(reinforcement_step=0.5))

    for _ in range(3):
        terrain.observe_trajectory(["a", "b", "sink"])

    assert terrain.transition_count(first) == 3
    assert terrain.transition_count(second) == 3
    assert terrain.influence(first) == 2.5
    assert terrain.influence(second) == 2.5

    # The retrieval overlay does not rewrite authoritative topology semantics.
    stored = topology.outgoing("a", {EdgeKind.CAUSAL})[0]
    assert stored.weight == 1.0
    assert stored.confidence == 1.0


def test_decay_reduces_influence_without_deleting_history() -> None:
    topology, first, _ = simple_topology()
    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(reinforcement_step=1.0, decay_rate=0.01),
    )
    terrain.reinforce(first)
    before = terrain.influence(first)
    nodes_before = set(topology.nodes)
    edges_before = list(topology.outgoing("a", {EdgeKind.CAUSAL}))

    terrain.decay(100.0)

    assert terrain.influence(first) < before
    assert set(topology.nodes) == nodes_before
    assert topology.outgoing("a", {EdgeKind.CAUSAL}) == edges_before


def test_scc_and_sink_attractor_discovery_is_deterministic() -> None:
    topology = CausalTopology()
    for node_id in ("x", "y", "sink"):
        topology.add_node(MemoryNode(id=node_id, text=node_id))
    topology.add_edge(causal("x", "y"))
    topology.add_edge(causal("y", "x"))
    topology.add_edge(causal("y", "sink"))

    terrain = DynamicTerrain(topology)
    terrain.observe_transition("x", "y")
    terrain.observe_transition("y", "x")
    terrain.observe_transition("y", "sink")

    assert terrain.strongly_connected_components() == (("sink",), ("x", "y"))
    assert terrain.sinks() == ("sink",)

    first = tuple(item.to_dict() for item in terrain.discover_attractors(register=True))
    second = tuple(item.to_dict() for item in terrain.discover_attractors(register=True))
    assert first == second
    assert any(item["origin"] == "discovered:sink" and item["node_id"] == "sink" for item in first)
    assert any(item["origin"] == "discovered:scc" and item["node_id"] == "x" for item in first)
    assert topology.attractor("sink").origin == "discovered:sink"
    assert topology.attractor("x").origin == "discovered:scc"


def test_discovery_never_overwrites_manual_attractor_provenance() -> None:
    topology, _, _ = simple_topology()
    topology.register_attractor("sink", confidence=0.9, origin="manual", metadata={"reason": "domain"})
    terrain = DynamicTerrain(topology)

    terrain.discover_attractors(register=True)

    descriptor = topology.attractor("sink")
    assert descriptor is not None
    assert descriptor.origin == "manual"
    assert descriptor.metadata == {"reason": "domain"}


def test_basin_drift_is_measurable_between_snapshots() -> None:
    topology, _, _ = simple_topology()
    topology.register_attractor("sink", origin="manual")
    terrain = DynamicTerrain(topology)

    before = terrain.snapshot()
    assert before.basins["sink"] == frozenset({"a", "b", "sink"})

    topology.add_node(MemoryNode(id="c", text="c"))
    topology.add_edge(causal("c", "sink"))
    after = terrain.snapshot()

    drift = terrain.basin_drift(before, after)
    assert drift.per_attractor["sink"] == 0.25
    assert drift.mean == 0.25


def test_protected_rare_edge_has_a_decay_floor_and_reset_preserves_history() -> None:
    topology, first, _ = simple_topology()
    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(reinforcement_step=1.0, decay_rate=1.0, protected_minimum_influence=0.8),
    )
    terrain.protect_edge(first)
    terrain.reinforce(first)
    terrain.decay(100)
    history = terrain.transition_counts

    assert terrain.influence(first) == 0.8
    assert terrain.protected_edges == frozenset({first.identity()})
    terrain.reset_navigation()
    assert terrain.influences == {}
    assert terrain.transition_counts == history


def test_surprise_signal_distinguishes_high_low_and_repetition() -> None:
    topology = CausalTopology()
    for node_id in ("root", "common", "rare"):
        topology.add_node(MemoryNode(id=node_id, text=node_id))
    common = causal("root", "common")
    rare = causal("root", "rare")
    topology.add_edge(common)
    topology.add_edge(rare)
    terrain = DynamicTerrain(topology, config=TerrainConfig(reinforcement_step=1.0))

    # Initially both alternatives are equally uncertain.
    initial = terrain.transition_surprise_signal(common)
    assert initial.source is SurpriseSource.TRANSITION_RESIDUAL
    assert initial.method_version == "laplace-outgoing-v1"
    assert initial.value == 0.5

    # Repeatedly observing the common transition makes it predictable.
    for _ in range(4):
        terrain.reinforce(common)
    common_signal = terrain.transition_surprise_signal(common)
    rare_signal = terrain.transition_surprise_signal(rare)
    assert common_signal.value < initial.value
    assert rare_signal.value > initial.value

    before = terrain.influence(rare)
    updated, used_signal = terrain.reinforce_by_transition_surprise(rare)
    assert used_signal.source is SurpriseSource.TRANSITION_RESIDUAL
    assert updated > before


def test_external_surprise_signal_requires_explicit_source() -> None:
    topology, first, _ = simple_topology()
    terrain = DynamicTerrain(topology, config=TerrainConfig(reinforcement_step=1.0))
    low = SurpriseSignal(value=0.1, source=SurpriseSource.OUTCOME_RESIDUAL, method_version="fixture-v1")
    high = SurpriseSignal(value=0.9, source=SurpriseSource.OUTCOME_RESIDUAL, method_version="fixture-v1")

    low_update = terrain.reinforce_by_surprise(first, signal=low)
    terrain.reset_navigation()
    high_update = terrain.reinforce_by_surprise(first, signal=high)
    assert high_update > low_update
