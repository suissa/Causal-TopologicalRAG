from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from ctrag import CausalProvenance, CausalTopology, EdgeKind, EventProjector, EventRecord


def _causal_edges(topology: CausalTopology) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for node_id in topology.nodes:
        for edge in topology.outgoing(node_id, {EdgeKind.CAUSAL}):
            result.add((edge.source, edge.target))
            assert edge.provenance is CausalProvenance.EVENT
            assert edge.evidence
            assert all(item.source == "event.causation_id" for item in edge.evidence)
    return result


def _adversarial_events(seed: int, *, explicit_causation: bool) -> tuple[list[EventRecord], set[tuple[str, str]]]:
    rng = random.Random(seed)
    count = 180
    base = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    ids = [f"evt-{index:03d}" for index in range(count)]
    expected: set[tuple[str, str]] = set()
    events: list[EventRecord] = []

    for index, event_id in enumerate(ids):
        # Deliberately make event-time unrelated to ingest order.
        event_time = base + timedelta(seconds=rng.randint(-10_000, 10_000))
        observed_at = base + timedelta(seconds=20_000 + index)

        execution_id = f"exec-{rng.randrange(6)}"
        correlation_id = f"corr-{rng.randrange(8)}"

        # These fields intentionally look causal but live only in payload/trace context.
        fake_parent = ids[rng.randrange(count)]
        fake_cause = ids[rng.randrange(count)]
        payload = {
            "trace_id": f"trace-{rng.randrange(5)}",
            "span_id": f"span-{index}",
            "parent_span_id": f"span-{rng.randrange(count)}",
            "traceparent": f"00-{rng.getrandbits(128):032x}-{rng.getrandbits(64):016x}-01",
            "parent_event_id": fake_parent,
            "causal_parent": fake_cause,
            "cause": fake_cause,
            "causation_id": fake_parent,
            "message": f"event {event_id} followed {fake_parent}",
        }

        causation_id: str | None = None
        if explicit_causation and index > 0 and index % 13 == 0:
            # Some parents will arrive before the child and some after it once shuffled.
            parent_index = (index * 17 + 23) % count
            if parent_index == index:
                parent_index = (parent_index + 1) % count
            causation_id = ids[parent_index]
            expected.add((causation_id, event_id))

        events.append(EventRecord(
            event_id=event_id,
            event_type="Chaos.Observed",
            timestamp=event_time,
            observed_at=observed_at,
            payload=payload,
            causation_id=causation_id,
            correlation_id=correlation_id,
            execution_id=execution_id,
            status=rng.choice(["ok", "retry", "timeout", "recovered"]),
        ))

    rng.shuffle(events)
    return events, expected


@pytest.mark.parametrize("seed", [7, 17, 29, 101])
def test_causal_chaos_never_promotes_trace_time_correlation_or_payload_hints(seed: int) -> None:
    """Adversarial telemetry without canonical causation must create zero CAUSAL edges.

    The fixture mixes:
    - shuffled arrival order;
    - inverted/random event time;
    - repeated execution and correlation identifiers;
    - trace/span parent relationships;
    - payload fields named cause/causation_id/parent_event_id;
    - messages that linguistically imply ordering.

    None of those signals has causal authority.
    """
    topology = CausalTopology()
    projector = EventProjector(topology)
    events, expected = _adversarial_events(seed, explicit_causation=False)

    assert expected == set()
    projector.ingest_many(events)

    assert _causal_edges(topology) == set()
    assert any(topology.outgoing(node_id, {EdgeKind.TEMPORAL}) for node_id in topology.nodes)
    assert any(topology.outgoing(node_id, {EdgeKind.BEHAVIORAL}) for node_id in topology.nodes)


@pytest.mark.parametrize("seed", [7, 17, 29, 101])
def test_causal_chaos_materializes_exactly_explicit_runtime_causation(seed: int) -> None:
    """With mixed true/false hints, the causal graph must equal the explicit declarations.

    This exercises pending out-of-order causation reconciliation as well as resistance
    to false causal hints from trace, temporal adjacency, correlation and payload text.
    """
    topology = CausalTopology()
    projector = EventProjector(topology)
    events, expected = _adversarial_events(seed, explicit_causation=True)

    projector.ingest_many(events)

    actual = _causal_edges(topology)
    assert actual == expected

    # Every declared child gets exactly one canonical causal parent in this fixture.
    expected_children = {child for _, child in expected}
    for child_id in expected_children:
        incoming = topology.incoming(child_id, {EdgeKind.CAUSAL})
        assert len(incoming) == 1
        assert incoming[0].evidence[0].source == "event.causation_id"


def test_causal_chaos_temporal_link_api_cannot_create_causal_authority() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)
    base = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)

    projector.ingest(EventRecord(
        event_id="deploy", event_type="Deploy.Done", timestamp=base, execution_id="a"
    ))
    projector.ingest(EventRecord(
        event_id="incident", event_type="Incident.Start",
        timestamp=base + timedelta(seconds=1), execution_id="b"
    ))

    from ctrag import TemporalScope

    projector.link_temporal("deploy", "incident", scope=TemporalScope.INCIDENT_WINDOW)

    assert topology.outgoing("deploy", {EdgeKind.CAUSAL}) == []
    temporal = topology.outgoing("deploy", {EdgeKind.TEMPORAL})
    assert len(temporal) == 1
    assert temporal[0].target == "incident"
