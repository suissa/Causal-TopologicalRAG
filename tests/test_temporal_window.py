from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    Edge,
    EdgeKind,
    MemoryNode,
    QueryMode,
    TemporalConsistencyWindow,
    TemporalScope,
)


def _node(node_id: str, at: datetime) -> MemoryNode:
    return MemoryNode(id=node_id, text=f"state {node_id}", timestamp=at)


def _temporal(source: str, target: str) -> Edge:
    return Edge(
        source,
        target,
        EdgeKind.TEMPORAL,
        temporal_scope=TemporalScope.EXECUTION,
    )


def test_temporal_consistency_window_rejects_large_event_time_gap() -> None:
    t0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    topology = CausalTopology()
    topology.add_node(_node("a", t0))
    topology.add_node(_node("b", t0 + timedelta(seconds=10)))
    topology.add_node(_node("c", t0 + timedelta(seconds=1000)))
    topology.add_edge(_temporal("a", "b"))
    topology.add_edge(_temporal("b", "c"))

    unconstrained = topology.distances(
        "a",
        direction="out",
        kinds={EdgeKind.TEMPORAL},
        max_hops=4,
    )
    constrained = topology.distances(
        "a",
        direction="out",
        kinds={EdgeKind.TEMPORAL},
        temporal_window=TemporalConsistencyWindow(max_gap_seconds=60),
        max_hops=4,
    )

    assert unconstrained == {"a": 0, "b": 1, "c": 2}
    assert constrained == {"a": 0, "b": 1}


def test_temporal_window_can_reject_or_allow_reverse_event_time() -> None:
    t0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    topology = CausalTopology()
    topology.add_node(_node("late-source", t0 + timedelta(seconds=5)))
    topology.add_node(_node("early-target", t0))
    topology.add_edge(_temporal("late-source", "early-target"))

    monotonic = topology.distances(
        "late-source",
        direction="out",
        kinds={EdgeKind.TEMPORAL},
        temporal_window=TemporalConsistencyWindow(
            max_gap_seconds=10,
            require_monotonic_event_time=True,
        ),
    )
    explicitly_non_monotonic = topology.distances(
        "late-source",
        direction="out",
        kinds={EdgeKind.TEMPORAL},
        temporal_window=TemporalConsistencyWindow(
            max_gap_seconds=10,
            require_monotonic_event_time=False,
        ),
    )

    assert monotonic == {"late-source": 0}
    assert explicitly_non_monotonic == {"late-source": 0, "early-target": 1}


def test_temporal_window_never_filters_or_promotes_causal_edges() -> None:
    t0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    topology = CausalTopology()
    topology.add_node(_node("cause", t0))
    topology.add_node(_node("effect", t0 + timedelta(days=30)))
    causal = Edge(
        "cause",
        "effect",
        EdgeKind.CAUSAL,
        provenance=CausalProvenance.EVENT,
    )
    topology.add_edge(causal)

    distances = topology.distances(
        "cause",
        direction="out",
        kinds={EdgeKind.CAUSAL},
        temporal_window=TemporalConsistencyWindow(max_gap_seconds=1),
    )

    assert distances == {"cause": 0, "effect": 1}
    assert topology.outgoing("cause", {EdgeKind.TEMPORAL}) == []
    assert topology.outgoing("cause", {EdgeKind.CAUSAL}) == [causal]


def test_retrieval_stage_respects_temporal_consistency_window() -> None:
    t0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    topology = CausalTopology()
    topology.add_node(_node("anchor", t0))
    topology.add_node(_node("near", t0 + timedelta(seconds=15)))
    topology.add_node(_node("far", t0 + timedelta(hours=2)))
    topology.add_edge(_temporal("anchor", "near"))
    topology.add_edge(_temporal("near", "far"))

    retriever = CTRetriever(topology)
    unconstrained = retriever.search_staged(
        "state",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["anchor"],
        k=3,
        max_hops=4,
    )
    constrained = retriever.search_staged(
        "state",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["anchor"],
        k=3,
        max_hops=4,
        temporal_window=TemporalConsistencyWindow(max_gap_seconds=60),
    )

    unconstrained_stage = next(
        stage for stage in unconstrained.stages if stage.name == "basin_topology_expansion"
    )
    constrained_stage = next(
        stage for stage in constrained.stages if stage.name == "basin_topology_expansion"
    )

    assert "far" in unconstrained_stage.node_ids
    assert "near" in constrained_stage.node_ids
    assert "far" not in constrained_stage.node_ids


def test_temporal_window_validation() -> None:
    try:
        TemporalConsistencyWindow(max_gap_seconds=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative max_gap_seconds must be rejected")
