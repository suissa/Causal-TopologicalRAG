from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CausalProvenance, Edge, EdgeEvidence, EdgeKind, MemoryNode
from .topology import CausalTopology


_MISSING = object()
_RESERVED_METADATA = {
    "event_type",
    "causation_id",
    "correlation_id",
    "execution_id",
    "intent_id",
    "actor_id",
    "action_id",
    "status",
    "payload",
    "_event_fingerprint",
}


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _lookup_path(raw: dict[str, Any], path: str, default: Any = _MISSING) -> Any:
    """Resolve a configurable dotted field path from an external event record."""

    current: Any = raw
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            if default is _MISSING:
                raise KeyError(path)
            return default
        current = current[part]
    return current


def _render_payload_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(slots=True, frozen=True)
class EventFieldMapping:
    """Map an external event schema into the canonical CT-RAG event contract.

    Mapping values are direct keys or dotted paths such as ``metadata.eventId``.
    """

    event_id: str = "event_id"
    event_type: str = "event_type"
    timestamp: str = "timestamp"
    payload: str = "payload"
    causation_id: str = "causation_id"
    correlation_id: str = "correlation_id"
    execution_id: str = "execution_id"
    intent_id: str = "intent_id"
    actor_id: str = "actor_id"
    action_id: str = "action_id"
    status: str = "status"

    def __post_init__(self) -> None:
        values = [
            self.event_id,
            self.event_type,
            self.timestamp,
            self.payload,
            self.causation_id,
            self.correlation_id,
            self.execution_id,
            self.intent_id,
            self.actor_id,
            self.action_id,
            self.status,
        ]
        for value in values:
            _require_non_empty(value, "field mapping path")
        if len(values) != len(set(values)):
            raise ValueError("event field mapping paths must be unique")


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

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.event_type, "event_type")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be an object/dict")
        try:
            json.dumps(self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload must be JSON-serializable") from exc

        for field_name in (
            "causation_id",
            "correlation_id",
            "execution_id",
            "intent_id",
            "actor_id",
            "action_id",
            "status",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty(value, field_name)
        if self.causation_id == self.event_id:
            raise ValueError("event cannot causally reference itself")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "actor_id": self.actor_id,
            "action_id": self.action_id,
            "status": self.status,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        mapping: EventFieldMapping | None = None,
    ) -> "EventRecord":
        if not isinstance(raw, dict):
            raise TypeError("event record must be a JSON object")
        mapping = mapping or EventFieldMapping()

        def required(path: str, canonical_name: str) -> Any:
            try:
                value = _lookup_path(raw, path)
            except KeyError as exc:
                raise ValueError(
                    f"missing required field {canonical_name!r} mapped from {path!r}"
                ) from exc
            if value is None:
                raise ValueError(
                    f"required field {canonical_name!r} mapped from {path!r} cannot be null"
                )
            return value

        event_id = required(mapping.event_id, "event_id")
        event_type = required(mapping.event_type, "event_type")
        timestamp_raw = required(mapping.timestamp, "timestamp")

        if not isinstance(timestamp_raw, str):
            raise TypeError("timestamp must be an ISO-8601 string")
        try:
            timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {timestamp_raw!r}") from exc

        payload = _lookup_path(raw, mapping.payload, {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object")

        def optional(path: str) -> str | None:
            value = _lookup_path(raw, path, None)
            return None if value is None else str(value)

        return cls(
            event_id=str(event_id),
            event_type=str(event_type),
            timestamp=timestamp,
            payload=dict(payload),
            causation_id=optional(mapping.causation_id),
            correlation_id=optional(mapping.correlation_id),
            execution_id=optional(mapping.execution_id),
            intent_id=optional(mapping.intent_id),
            actor_id=optional(mapping.actor_id),
            action_id=optional(mapping.action_id),
            status=optional(mapping.status),
        )


class EventProjector:
    """Project authoritative event history into a CT-RAG retrieval topology.

    The projector never promotes order to causality. Explicit ``causation_id``
    becomes an EVENT-provenance causal edge. Sequential events in one execution
    receive temporal and behavioral relations only.
    """

    def __init__(self, topology: CausalTopology) -> None:
        self.topology = topology
        self._last_by_execution: dict[str, str] = {}
        self._pending_causation: dict[str, set[str]] = {}
        self._bootstrap_execution_tails()

    def _bootstrap_execution_tails(self) -> None:
        """Recover deterministic execution tails from an already-loaded topology."""

        by_execution: dict[str, set[str]] = {}
        for node_id, node in self.topology.nodes.items():
            execution_id = node.metadata.get("execution_id")
            if isinstance(execution_id, str) and execution_id:
                by_execution.setdefault(execution_id, set()).add(node_id)

        for execution_id, node_ids in by_execution.items():
            non_tails: set[str] = set()
            for node_id in node_ids:
                for edge in self.topology.outgoing(node_id, {EdgeKind.TEMPORAL}):
                    if edge.target in node_ids:
                        non_tails.add(node_id)
            tails = node_ids.difference(non_tails)
            if tails:
                self._last_by_execution[execution_id] = max(
                    tails,
                    key=lambda node_id: (self.topology.nodes[node_id].timestamp, node_id),
                )

    def _metadata_for(self, event: EventRecord) -> dict[str, Any]:
        # Payload remains available both as a nested authoritative value and as
        # non-reserved convenience metadata. Canonical identifiers always win.
        metadata = {
            key: value
            for key, value in event.payload.items()
            if key not in _RESERVED_METADATA
        }
        metadata.update({
            "event_type": event.event_type,
            "causation_id": event.causation_id,
            "correlation_id": event.correlation_id,
            "execution_id": event.execution_id,
            "intent_id": event.intent_id,
            "actor_id": event.actor_id,
            "action_id": event.action_id,
            "status": event.status,
            "payload": dict(event.payload),
            "_event_fingerprint": event.fingerprint(),
        })
        return {key: value for key, value in metadata.items() if value is not None}

    def _node_for(self, event: EventRecord) -> MemoryNode:
        text_parts = [event.event_type]
        if event.status:
            text_parts.append(f"status={event.status}")
        for key, value in sorted(event.payload.items()):
            text_parts.append(f"{key}={_render_payload_value(value)}")
        return MemoryNode(
            id=event.event_id,
            text=" ".join(text_parts),
            timestamp=event.timestamp,
            metadata=self._metadata_for(event),
        )

    def _ensure_edge(self, edge: Edge) -> None:
        if not self.topology.has_edge(edge):
            self.topology.add_edge(edge)

    def _causal_edge(self, parent_id: str, child_id: str) -> Edge:
        return Edge(
            source=parent_id,
            target=child_id,
            kind=EdgeKind.CAUSAL,
            provenance=CausalProvenance.EVENT,
            confidence=1.0,
            evidence=(EdgeEvidence(
                id=f"causation:{child_id}",
                source="event.causation_id",
                metadata={"child_event_id": child_id, "parent_event_id": parent_id},
            ),),
        )

    def _project_causation(self, event: EventRecord) -> None:
        if not event.causation_id:
            return
        if event.causation_id in self.topology.nodes:
            self._ensure_edge(self._causal_edge(event.causation_id, event.event_id))
        else:
            self._pending_causation.setdefault(event.causation_id, set()).add(event.event_id)

    def _resolve_pending_children(self, parent_id: str) -> None:
        for child_id in sorted(self._pending_causation.pop(parent_id, set())):
            if child_id in self.topology.nodes:
                self._ensure_edge(self._causal_edge(parent_id, child_id))

    def _project_execution_sequence(self, event: EventRecord) -> None:
        if not event.execution_id:
            return
        previous = self._last_by_execution.get(event.execution_id)
        if previous and previous != event.event_id:
            self._ensure_edge(Edge(
                source=previous,
                target=event.event_id,
                kind=EdgeKind.TEMPORAL,
            ))
            self._ensure_edge(Edge(
                source=previous,
                target=event.event_id,
                kind=EdgeKind.BEHAVIORAL,
            ))
        self._last_by_execution[event.execution_id] = event.event_id

    def ingest(self, event: EventRecord) -> MemoryNode:
        existing = self.topology.nodes.get(event.event_id)
        if existing is not None:
            existing_fingerprint = existing.metadata.get("_event_fingerprint")
            if existing_fingerprint == event.fingerprint():
                return existing
            raise ValueError(
                f"event id conflict for {event.event_id!r}: existing projection differs from replay"
            )

        node = self._node_for(event)
        self.topology.add_node(node)
        self._project_causation(event)
        self._resolve_pending_children(event.event_id)
        self._project_execution_sequence(event)
        return node

    def ingest_many(self, events: list[EventRecord]) -> list[MemoryNode]:
        return [self.ingest(event) for event in events]

    def ingest_ndjson(
        self,
        path: str | Path,
        *,
        mapping: EventFieldMapping | None = None,
    ) -> list[MemoryNode]:
        nodes: list[MemoryNode] = []
        mapping = mapping or EventFieldMapping()
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at line {line_number}: {exc.msg}"
                    ) from exc

                event_identity: Any = "<unknown>"
                if isinstance(raw, dict):
                    event_identity = _lookup_path(raw, mapping.event_id, "<unknown>")
                try:
                    event = EventRecord.from_dict(raw, mapping=mapping)
                    nodes.append(self.ingest(event))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid event at line {line_number} "
                        f"(event_id={event_identity!r}): {exc}"
                    ) from exc
        return nodes
