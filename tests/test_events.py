from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ctrag import (
    CausalTopology,
    EdgeKind,
    EventFieldMapping,
    EventProjector,
    EventRecord,
)


FIXTURE = Path(__file__).parent / "fixtures" / "multi_step_trace.ndjson"


def test_explicit_causation_reconciles_when_parent_arrives_late() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)
    now = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)

    child = EventRecord(
        event_id="child",
        event_type="Child",
        timestamp=now,
        causation_id="parent",
    )
    parent = EventRecord(
        event_id="parent",
        event_type="Parent",
        timestamp=now,
    )

    projector.ingest(child)
    assert topology.incoming("child", {EdgeKind.CAUSAL}) == []

    projector.ingest(parent)
    causal = topology.incoming("child", {EdgeKind.CAUSAL})
    assert len(causal) == 1
    assert causal[0].source == "parent"
    assert causal[0].evidence[0].source == "event.causation_id"


def test_same_execution_sequence_never_implies_causality() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)
    now = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)

    projector.ingest(EventRecord(
        event_id="e1",
        event_type="Step.One",
        timestamp=now,
        execution_id="exec-1",
    ))
    projector.ingest(EventRecord(
        event_id="e2",
        event_type="Step.Two",
        timestamp=now.replace(second=1),
        execution_id="exec-1",
    ))

    assert len(topology.outgoing("e1", {EdgeKind.TEMPORAL})) == 1
    assert len(topology.outgoing("e1", {EdgeKind.BEHAVIORAL})) == 1
    assert topology.outgoing("e1", {EdgeKind.CAUSAL}) == []


def test_duplicate_replay_is_idempotent_and_conflicting_reuse_fails() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)
    event = EventRecord(
        event_id="e1",
        event_type="Stable.Event",
        timestamp=datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc),
        payload={"value": 1},
    )

    first = projector.ingest(event)
    second = projector.ingest(EventRecord.from_dict(event.to_dict()))

    assert first is second
    assert list(topology.nodes) == ["e1"]

    conflicting = EventRecord(
        event_id="e1",
        event_type="Stable.Event",
        timestamp=event.timestamp,
        payload={"value": 2},
    )
    with pytest.raises(ValueError, match="event id conflict"):
        projector.ingest(conflicting)


def test_nested_field_mapping_preserves_canonical_identifiers() -> None:
    mapping = EventFieldMapping(
        event_id="meta.id",
        event_type="meta.type",
        timestamp="meta.at",
        payload="data",
        causation_id="links.cause",
        correlation_id="links.correlation",
        execution_id="scope.execution",
        intent_id="scope.intent",
        actor_id="scope.actor",
        action_id="scope.action",
        status="meta.status",
    )
    raw = {
        "meta": {
            "id": "evt-1",
            "type": "Payment.Authorized",
            "at": "2026-09-09T08:00:00Z",
            "status": "ok",
        },
        "links": {"cause": "evt-0", "correlation": "corr-1"},
        "scope": {
            "execution": "exec-1",
            "intent": "checkout",
            "actor": "actor-1",
            "action": "payment",
        },
        "data": {"amount": 42},
    }

    event = EventRecord.from_dict(raw, mapping=mapping)
    assert event.event_id == "evt-1"
    assert event.causation_id == "evt-0"
    assert event.execution_id == "exec-1"
    assert event.payload == {"amount": 42}


def test_invalid_ndjson_reports_line_and_event_identity(tmp_path: Path) -> None:
    path = tmp_path / "invalid.ndjson"
    path.write_text(
        '{"event_id":"ok","event_type":"Ok","timestamp":"2026-09-09T08:00:00Z"}\n'
        '{"event_id":"broken","event_type":"Broken","timestamp":"not-a-date"}\n',
        encoding="utf-8",
    )

    projector = EventProjector(CausalTopology())
    with pytest.raises(ValueError) as exc_info:
        projector.ingest_ndjson(path)

    message = str(exc_info.value)
    assert "line 2" in message
    assert "event_id='broken'" in message


def test_multi_step_fixture_reconstructs_causal_and_execution_trace() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)

    nodes = projector.ingest_ndjson(FIXTURE)
    assert [node.id for node in nodes] == ["e1", "e2", "e3", "e4", "e5"]

    causal_pairs = {
        (edge.source, edge.target)
        for node_id in topology.nodes
        for edge in topology.outgoing(node_id, {EdgeKind.CAUSAL})
    }
    temporal_pairs = {
        (edge.source, edge.target)
        for node_id in topology.nodes
        for edge in topology.outgoing(node_id, {EdgeKind.TEMPORAL})
    }
    behavioral_pairs = {
        (edge.source, edge.target)
        for node_id in topology.nodes
        for edge in topology.outgoing(node_id, {EdgeKind.BEHAVIORAL})
    }

    expected = {("e1", "e2"), ("e2", "e3"), ("e3", "e4"), ("e4", "e5")}
    assert causal_pairs == expected
    assert temporal_pairs == expected
    assert behavioral_pairs == expected

    path = topology.causal_path_evidence("e5", "e1", direction="in", max_hops=8)
    assert path is not None
    assert path.nodes == ("e5", "e4", "e3", "e2", "e1")
    assert path.hops == 4


def test_fixture_replay_is_idempotent() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)

    projector.ingest_ndjson(FIXTURE)
    initial_edge_counts = {
        kind: sum(len(topology.outgoing(node_id, {kind})) for node_id in topology.nodes)
        for kind in (EdgeKind.CAUSAL, EdgeKind.TEMPORAL, EdgeKind.BEHAVIORAL)
    }

    projector.ingest_ndjson(FIXTURE)
    replay_edge_counts = {
        kind: sum(len(topology.outgoing(node_id, {kind})) for node_id in topology.nodes)
        for kind in (EdgeKind.CAUSAL, EdgeKind.TEMPORAL, EdgeKind.BEHAVIORAL)
    }

    assert replay_edge_counts == initial_edge_counts
    assert len(topology.nodes) == 5
