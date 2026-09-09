from datetime import datetime, timedelta, timezone

import pytest

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    Edge,
    EdgeEvidence,
    EdgeKind,
    EventProjector,
    EventRecord,
    MemoryNode,
    QueryMode,
    RetrievalWeights,
)


def test_why_query_prefers_explicit_causal_ancestor() -> None:
    topology = CausalTopology()
    now = datetime.now(timezone.utc)

    topology.add_node(MemoryNode(
        id="cause",
        text="payment provider authorized transaction",
        timestamp=now,
        metadata={"execution_id": "exec-1", "intent_id": "checkout"},
    ))
    topology.add_node(MemoryNode(
        id="error",
        text="inventory reservation failed because stock changed",
        timestamp=now + timedelta(seconds=1),
        metadata={"execution_id": "exec-1", "intent_id": "checkout"},
    ))
    topology.add_node(MemoryNode(
        id="lookalike",
        text="inventory reservation failure troubleshooting stock changed",
        timestamp=now + timedelta(seconds=2),
        metadata={"execution_id": "exec-other", "intent_id": "docs"},
    ))
    topology.add_edge(Edge(
        source="cause",
        target="error",
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
        confidence=1.0,
    ))

    retriever = CTRetriever(topology)
    hits = retriever.search(
        "inventory reservation failed stock changed",
        mode=QueryMode.WHY,
        anchor_ids=["error"],
        k=2,
    )

    assert hits[0].node.id == "cause"
    assert hits[0].components["causal"] > 0
    assert hits[0].causal_hops == 1


def test_basin_is_reverse_reachable_from_attractor() -> None:
    topology = CausalTopology()
    for node_id in ("a", "b", "c", "sink", "unrelated"):
        topology.add_node(MemoryNode(id=node_id, text=node_id))
    for source, target in (("a", "b"), ("b", "c"), ("c", "sink")):
        topology.add_edge(Edge(
            source=source,
            target=target,
            kind=EdgeKind.CAUSAL,
            provenance=CausalProvenance.EXECUTION,
        ))
    topology.register_attractor("sink")

    assert topology.basin("sink") == {"a", "b", "c", "sink"}
    assert topology.attractors_for("a") == {"sink"}
    assert topology.attractors_for("unrelated") == set()


def test_event_projector_does_not_infer_causality_from_order() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)
    now = datetime.now(timezone.utc)

    projector.ingest(EventRecord(
        event_id="e1",
        event_type="Step.One",
        timestamp=now,
        execution_id="exec-1",
    ))
    projector.ingest(EventRecord(
        event_id="e2",
        event_type="Step.Two",
        timestamp=now + timedelta(seconds=1),
        execution_id="exec-1",
    ))

    assert topology.outgoing("e1", {EdgeKind.TEMPORAL})
    assert topology.outgoing("e1", {EdgeKind.BEHAVIORAL})
    assert topology.outgoing("e1", {EdgeKind.CAUSAL}) == []


def test_event_projector_preserves_explicit_causation() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)

    projector.ingest(EventRecord(event_id="e1", event_type="Action.Ok"))
    projector.ingest(EventRecord(
        event_id="e2",
        event_type="Next.Started",
        causation_id="e1",
    ))

    causal_edges = topology.outgoing("e1", {EdgeKind.CAUSAL})
    assert len(causal_edges) == 1
    assert causal_edges[0].target == "e2"
    assert causal_edges[0].provenance is CausalProvenance.EVENT


def test_memory_node_round_trip_preserves_identity_metadata_and_embedding() -> None:
    node = MemoryNode(
        id="event-42",
        text="Inventory reservation failed",
        timestamp=datetime(2026, 9, 9, 7, 0, tzinfo=timezone.utc),
        metadata={"execution_id": "exec-7", "nested": {"retry": 2}},
        embedding=(0.1, -0.2, 0.3),
        is_attractor=True,
    )

    restored = MemoryNode.from_dict(node.to_dict())

    assert restored.to_dict() == node.to_dict()


def test_causal_edge_round_trip_preserves_evidence_and_provenance_metadata() -> None:
    edge = Edge(
        source="a",
        target="b",
        kind=EdgeKind.CAUSAL,
        weight=0.8,
        provenance=CausalProvenance.EXECUTION,
        confidence=0.95,
        evidence=(EdgeEvidence(
            id="trace-17",
            source="event-store",
            metadata={"stream": "checkout-1", "revision": 4},
        ),),
        provenance_metadata={"collector": "event-projector", "schema": 1},
    )

    restored = Edge.from_dict(edge.to_dict())

    assert restored.to_dict() == edge.to_dict()
    assert restored.identity() == edge.identity()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MemoryNode(id="", text="x"),
        lambda: MemoryNode(id="x", text="x", timestamp=datetime(2026, 1, 1)),
        lambda: Edge(source="", target="b", kind=EdgeKind.TEMPORAL),
        lambda: Edge(source="a", target="b", kind=EdgeKind.CAUSAL),
        lambda: Edge(
            source="a",
            target="b",
            kind=EdgeKind.TEMPORAL,
            provenance=CausalProvenance.EVENT,
        ),
        lambda: Edge(source="a", target="b", kind=EdgeKind.CAUSAL, provenance=CausalProvenance.EVENT, confidence=float("nan")),
        lambda: Edge(source="a", target="b", kind=EdgeKind.TEMPORAL, weight=float("inf")),
        lambda: Edge(
            source="a",
            target="b",
            kind=EdgeKind.TEMPORAL,
            evidence=(EdgeEvidence(id="evidence"),),
        ),
        lambda: RetrievalWeights(-0.1, 0.2, 0.2, 0.2, 0.2, 0.3),
    ],
)
def test_invalid_model_combinations_fail_fast(factory) -> None:
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_duplicate_edge_identity_is_rejected() -> None:
    topology = CausalTopology()
    topology.add_node(MemoryNode(id="a", text="a"))
    topology.add_node(MemoryNode(id="b", text="b"))
    edge = Edge(
        source="a",
        target="b",
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    )

    topology.add_edge(edge)

    with pytest.raises(ValueError, match="edge already exists"):
        topology.add_edge(Edge.from_dict(edge.to_dict()))


def test_causal_and_temporal_edges_keep_distinct_identity() -> None:
    topology = CausalTopology()
    topology.add_node(MemoryNode(id="a", text="a"))
    topology.add_node(MemoryNode(id="b", text="b"))
    topology.add_edge(Edge(
        source="a",
        target="b",
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    topology.add_edge(Edge(source="a", target="b", kind=EdgeKind.TEMPORAL))

    assert len(topology.outgoing("a", {EdgeKind.CAUSAL})) == 1
    assert len(topology.outgoing("a", {EdgeKind.TEMPORAL})) == 1
