from datetime import datetime, timezone

import pytest

from ctrag import (
    AttractorDescriptor,
    CausalProvenance,
    CausalTopology,
    Edge,
    EdgeKind,
    MemoryNode,
)


def _node(node_id: str) -> MemoryNode:
    return MemoryNode(id=node_id, text=node_id, timestamp=datetime(2026, 9, 9, tzinfo=timezone.utc))


def _causal(topology: CausalTopology, source: str, target: str) -> None:
    topology.add_edge(Edge(
        source=source,
        target=target,
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))


def test_attractor_descriptor_is_first_class_and_round_trippable() -> None:
    topology = CausalTopology()
    topology.add_node(_node("sink"))

    descriptor = topology.register_attractor(
        "sink",
        confidence=0.85,
        origin="manual-test",
        metadata={"outcome": "recovered"},
    )

    assert topology.attractor("sink") == descriptor
    assert topology.nodes["sink"].is_attractor is True
    assert AttractorDescriptor.from_dict(descriptor.to_dict()) == descriptor


def test_branching_basins_have_memberships_affinity_boundaries_and_neighbors() -> None:
    topology = CausalTopology()
    for node_id in ("root", "split", "success", "failure"):
        topology.add_node(_node(node_id))
    _causal(topology, "root", "split")
    _causal(topology, "split", "success")
    _causal(topology, "split", "failure")
    topology.register_attractor("success", metadata={"status": "ok"})
    topology.register_attractor("failure", confidence=0.9, metadata={"status": "error"})

    assert topology.basin("success") == {"root", "split", "success"}
    assert topology.basin("failure") == {"root", "split", "failure"}
    assert topology.basin_memberships("split") == {"success", "failure"}

    affinity = topology.shared_basin_affinity("root", "split")
    assert affinity.shared_attractors == ("failure", "success")
    assert affinity.score == pytest.approx(0.6)

    assert topology.basin_boundary("success") == {"split"}
    assert topology.neighboring_basins("success") == {"failure"}
    assert topology.neighboring_basins("failure") == {"success"}


def test_unrelated_component_does_not_leak_into_basin() -> None:
    topology = CausalTopology()
    for node_id in ("a", "sink", "x", "other"):
        topology.add_node(_node(node_id))
    _causal(topology, "a", "sink")
    _causal(topology, "x", "other")
    topology.register_attractor("sink")
    topology.register_attractor("other")

    assert topology.basin("sink") == {"a", "sink"}
    assert topology.basin_memberships("x") == {"other"}
    assert topology.shared_basin_affinity("a", "x").score == 0.0
    assert topology.neighboring_basins("sink") == set()


def test_cyclic_region_converging_to_attractor_remains_finite() -> None:
    topology = CausalTopology()
    for node_id in ("a", "b", "sink"):
        topology.add_node(_node(node_id))
    _causal(topology, "a", "b")
    _causal(topology, "b", "a")
    _causal(topology, "b", "sink")
    topology.register_attractor("sink")

    assert topology.basin("sink") == {"a", "b", "sink"}
    assert topology.basin_memberships("a") == {"sink"}


def test_invalid_attractor_confidence_fails_fast() -> None:
    topology = CausalTopology()
    topology.add_node(_node("sink"))

    with pytest.raises(ValueError):
        topology.register_attractor("sink", confidence=1.1)
