from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ctrag.models import Edge, EdgeKind, MemoryNode, TemporalScope
from ctrag.topology import CausalTopology


@dataclass(frozen=True, slots=True)
class ExternalDatasetProfile:
    name: str
    source_url: str
    evidence_shapes: tuple[str, ...]
    supports_execution_topology: bool
    supports_causal_ground_truth: bool
    intended_use: str


AIOPS2020_PROFILE = ExternalDatasetProfile(
    name="AIOps2020",
    source_url="https://github.com/NetManAIOps",
    evidence_shapes=("TRACE", "METRIC_SERIES"),
    supports_execution_topology=True,
    supports_causal_ground_truth=False,
    intended_use="external trace/telemetry validity and root-cause localization when source labels are available",
)

SMD_PROFILE = ExternalDatasetProfile(
    name="SMD",
    source_url="https://github.com/NetManAIOps/OmniAnomaly",
    evidence_shapes=("METRIC_SERIES",),
    supports_execution_topology=False,
    supports_causal_ground_truth=False,
    intended_use="external anomaly/drift and null-baseline validation; not causal-path evaluation",
)


@dataclass(frozen=True, slots=True)
class TraceCSVMapping:
    timestamp: str = "timestamp"
    trace_id: str = "trace_id"
    span_id: str = "span_id"
    parent_span_id: str = "parent_span_id"
    service: str = "service"
    operation: str = "operation"
    status: str = "status"
    duration_ms: str = "duration_ms"


def _parse_timestamp(raw: str) -> datetime:
    value = raw.strip()
    if not value:
        raise ValueError("trace timestamp cannot be empty")
    try:
        numeric = float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    # Accept seconds or millisecond Unix timestamps.
    if numeric > 10_000_000_000:
        numeric /= 1000.0
    return datetime.fromtimestamp(numeric, tz=timezone.utc)


def load_trace_csv(
    path: Path,
    *,
    mapping: TraceCSVMapping | None = None,
    source_name: str = "external_trace_csv",
) -> CausalTopology:
    """Project external trace CSV into non-causal execution topology.

    Parent/child span relations become BEHAVIORAL edges. Consecutive spans in a
    trace become TEMPORAL edges. No CAUSAL edge is created by this adapter.
    """
    mapping = mapping or TraceCSVMapping()
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows.append({key: (value or "") for key, value in row.items()})

    topology = CausalTopology()
    by_trace: dict[str, list[tuple[datetime, str]]] = {}
    parent_links: list[tuple[str, str]] = []

    for row in rows:
        trace_id = row[mapping.trace_id].strip()
        span_id = row[mapping.span_id].strip()
        if not trace_id or not span_id:
            raise ValueError("trace_id and span_id are required")
        timestamp = _parse_timestamp(row[mapping.timestamp])
        node_id = f"trace:{trace_id}:{span_id}"
        operation = row.get(mapping.operation, "").strip()
        service = row.get(mapping.service, "").strip()
        status = row.get(mapping.status, "").strip()
        duration = row.get(mapping.duration_ms, "").strip()
        topology.add_node(MemoryNode(
            id=node_id,
            text=" ".join(part for part in (service, operation, status) if part),
            timestamp=timestamp,
            metadata={
                "execution_id": trace_id,
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": row.get(mapping.parent_span_id, "").strip() or None,
                "service": service,
                "operation": operation,
                "status": status,
                "duration_ms": duration,
                "evidence_shape": "TRACE",
                "source": source_name,
                "causality": "unknown",
            },
        ))
        by_trace.setdefault(trace_id, []).append((timestamp, node_id))
        parent = row.get(mapping.parent_span_id, "").strip()
        if parent:
            parent_links.append((f"trace:{trace_id}:{parent}", node_id))

    for trace_id, members in by_trace.items():
        ordered = [node_id for _, node_id in sorted(members, key=lambda item: (item[0], item[1]))]
        for left, right in zip(ordered, ordered[1:]):
            edge = Edge(left, right, EdgeKind.TEMPORAL, temporal_scope=TemporalScope.EXECUTION)
            if not topology.has_edge(edge):
                topology.add_edge(edge)

    for parent, child in parent_links:
        if parent in topology.nodes and child in topology.nodes:
            edge = Edge(parent, child, EdgeKind.BEHAVIORAL)
            if not topology.has_edge(edge):
                topology.add_edge(edge)

    return topology


def capability_matrix() -> dict[str, dict[str, object]]:
    profiles = (AIOPS2020_PROFILE, SMD_PROFILE)
    return {
        profile.name: {
            "source_url": profile.source_url,
            "evidence_shapes": list(profile.evidence_shapes),
            "supports_execution_topology": profile.supports_execution_topology,
            "supports_causal_ground_truth": profile.supports_causal_ground_truth,
            "intended_use": profile.intended_use,
        }
        for profile in profiles
    }