from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CausalProvenance, Edge, EdgeEvidence, EdgeKind, MemoryNode, TemporalScope
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
    "event_time",
    "observed_at",
    "valid_start",
    "valid_end",
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
    observed_at: str = "observed_at"
    valid_start: str = "valid_start"
    valid_end: str = "valid_end"
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
            self.observed_at,
            self.valid_start,
            self.valid_end,
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
    observed_at: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    causation_id: str | None = None
    correlation_id: str | None = None
    execution_id: str | None = None
    intent_id: str | None = None
    actor_id: str | None = None
    action_id: str | None = None
    status: str | None = None
    valid_start: datetime | None = None
    valid_end: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.event_type, "event_type")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.observed_at is not None:
            if not isinstance(self.observed_at, datetime):
                raise TypeError("observed_at must be a datetime")
            if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
                raise ValueError("observed_at must be timezone-aware")
        for field_name, value in (("valid_start", self.valid_start), ("valid_end", self.valid_end)):
            if value is not None:
                if not isinstance(value, datetime):
                    raise TypeError(f"{field_name} must be a datetime")
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{field_name} must be timezone-aware")
        if self.valid_start is not None and self.valid_end is not None:
            if self.valid_end < self.valid_start:
                raise ValueError("valid_end must be greater than or equal to valid_start")
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

    @property
    def event_time(self) -> datetime:
        """Authoritative time at which the source says the event occurred."""
        return self.timestamp

    def to_dict(self) -> dict[str, Any]:
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "observed_at": None if self.observed_at is None else self.observed_at.isoformat(),
            "payload": dict(self.payload),
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "actor_id": self.actor_id,
            "action_id": self.action_id,
            "status": self.status,
        }
        if self.valid_start is not None:
            result["valid_start"] = self.valid_start.isoformat()
        if self.valid_end is not None:
            result["valid_end"] = self.valid_end.isoformat()
        return result

    def fingerprint(self) -> str:
        # observed_at is transaction/ingest metadata, not part of the event's
        # authoritative identity. Replaying the same event at a later time must
        # remain idempotent.
        canonical_event = self.to_dict()
        canonical_event.pop("observed_at", None)
        canonical = json.dumps(
            canonical_event,
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

        def optional_datetime(path: str, canonical_name: str) -> datetime | None:
            value = _lookup_path(raw, path, None)
            if value is None:
                return None
            if not isinstance(value, str):
                raise TypeError(f"{canonical_name} must be an ISO-8601 string")
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid ISO-8601 {canonical_name}: {value!r}") from exc

        return cls(
            event_id=str(event_id),
            event_type=str(event_type),
            timestamp=timestamp,
            observed_at=optional_datetime(mapping.observed_at, "observed_at"),
            valid_start=optional_datetime(mapping.valid_start, "valid_start"),
            valid_end=optional_datetime(mapping.valid_end, "valid_end"),
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

    Temporal topology is intentionally execution-scoped: TEMPORAL edges connect
    consecutive projected events sharing the same ``execution_id``. They are not
    global-stream adjacency, so unrelated system load cannot change temporal-hop
    counts inside an execution.

    The projector never promotes order to causality. Explicit ``causation_id``
    becomes an EVENT-provenance causal edge. Sequential events in one execution
    receive temporal and behavioral relations only.
    """

    TEMPORAL_SCOPE = "execution_id"

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

    def _metadata_for(self, event: EventRecord, observed_at: datetime) -> dict[str, Any]:
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
            "event_time": event.timestamp.isoformat(),
            "observed_at": observed_at.isoformat(),
            "valid_start": None if event.valid_start is None else event.valid_start.isoformat(),
            "valid_end": None if event.valid_end is None else event.valid_end.isoformat(),
            "payload": dict(event.payload),
            "_event_fingerprint": event.fingerprint(),
        })
        return {key: value for key, value in metadata.items() if value is not None}

    def _node_for(self, event: EventRecord, observed_at: datetime) -> MemoryNode:
        text_parts = [event.event_type]
        if event.status:
            text_parts.append(f"status={event.status}")
        for key, value in sorted(event.payload.items()):
            text_parts.append(f"{key}={_render_payload_value(value)}")
        return MemoryNode(
            id=event.event_id,
            text=" ".join(text_parts),
            timestamp=event.timestamp,
            metadata=self._metadata_for(event, observed_at),
            valid_start=event.valid_start,
            valid_end=event.valid_end,
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
                temporal_scope=TemporalScope.EXECUTION,
            ))
            self._ensure_edge(Edge(
                source=previous,
                target=event.event_id,
                kind=EdgeKind.BEHAVIORAL,
            ))
        self._last_by_execution[event.execution_id] = event.event_id

    def link_temporal(
        self,
        source_id: str,
        target_id: str,
        *,
        scope: TemporalScope,
    ) -> Edge:
        """Link already-projected records with explicit temporal scope.

        This is the public cross-execution temporal relation API for relations
        such as deploy-before-execution or config-change-before-incident.
        It never creates causal authority.
        """
        if source_id not in self.topology.nodes:
            raise KeyError(source_id)
        if target_id not in self.topology.nodes:
            raise KeyError(target_id)
        if source_id == target_id:
            raise ValueError("temporal relation endpoints must be distinct")
        edge = Edge(
            source=source_id,
            target=target_id,
            kind=EdgeKind.TEMPORAL,
            temporal_scope=scope,
        )
        self._ensure_edge(edge)
        return edge

    def ingest(self, event: EventRecord) -> MemoryNode:
        existing = self.topology.nodes.get(event.event_id)
        if existing is not None:
            existing_fingerprint = existing.metadata.get("_event_fingerprint")
            if existing_fingerprint == event.fingerprint():
                return existing
            raise ValueError(
                f"event id conflict for {event.event_id!r}: existing projection differs from replay"
            )

        observed_at = event.observed_at or datetime.now(timezone.utc)
        node = self._node_for(event, observed_at)
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
