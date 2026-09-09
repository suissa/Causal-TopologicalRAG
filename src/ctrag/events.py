from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CausalProvenance, Edge, EdgeKind, MemoryNode
from .topology import CausalTopology


@dataclass(slots=True)
class EventRecord:
    event_id: str
    event_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = field(default_factory=dict)
    causation_id: str | None = None
    correlation_id: str | None = None
    execution_id: str | None = None
    intent_id: str | None = None
    actor_id: str | None = None
    action_id: str | None = None
    status: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EventRecord":
        timestamp = raw.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
        return cls(
            event_id=str(raw["event_id"]),
            event_type=str(raw["event_type"]),
            timestamp=timestamp,
            payload=dict(raw.get("payload") or {}),
            causation_id=raw.get("causation_id"),
            correlation_id=raw.get("correlation_id"),
            execution_id=raw.get("execution_id"),
            intent_id=raw.get("intent_id"),
            actor_id=raw.get("actor_id"),
            action_id=raw.get("action_id"),
            status=raw.get("status"),
        )


class EventProjector:
    """Project event-sourced history into a CT-RAG topology.

    Explicit ``causation_id`` becomes a causal edge when it references a known
    event. Sequential events in one execution get temporal + behavioral edges,
    never implicit causal edges.
    """

    def __init__(self, topology: CausalTopology) -> None:
        self.topology = topology
        self._last_by_execution: dict[str, str] = {}

    def ingest(self, event: EventRecord) -> MemoryNode:
        metadata = {
            "event_type": event.event_type,
            "causation_id": event.causation_id,
            "correlation_id": event.correlation_id,
            "execution_id": event.execution_id,
            "intent_id": event.intent_id,
            "actor_id": event.actor_id,
            "action_id": event.action_id,
            "status": event.status,
            **event.payload,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        text_parts = [event.event_type]
        if event.status:
            text_parts.append(f"status={event.status}")
        if event.payload:
            text_parts.extend(f"{key}={value}" for key, value in sorted(event.payload.items()))

        node = MemoryNode(
            id=event.event_id,
            text=" ".join(text_parts),
            timestamp=event.timestamp,
            metadata=metadata,
        )
        self.topology.add_node(node)

        if event.causation_id and event.causation_id in self.topology.nodes:
            self.topology.add_edge(
                Edge(
                    source=event.causation_id,
                    target=event.event_id,
                    kind=EdgeKind.CAUSAL,
                    provenance=CausalProvenance.EVENT,
                    confidence=1.0,
                )
            )

        if event.execution_id:
            previous = self._last_by_execution.get(event.execution_id)
            if previous and previous != event.event_id:
                self.topology.add_edge(
                    Edge(source=previous, target=event.event_id, kind=EdgeKind.TEMPORAL)
                )
                self.topology.add_edge(
                    Edge(source=previous, target=event.event_id, kind=EdgeKind.BEHAVIORAL)
                )
            self._last_by_execution[event.execution_id] = event.event_id

        return node

    def ingest_ndjson(self, path: str | Path) -> list[MemoryNode]:
        nodes: list[MemoryNode] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at line {line_number}") from exc
                nodes.append(self.ingest(EventRecord.from_dict(raw)))
        return nodes
