from datetime import datetime, timedelta, timezone

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    Edge,
    EdgeKind,
    EventProjector,
    EventRecord,
    MemoryNode,
    QueryMode,
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
