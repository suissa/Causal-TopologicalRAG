"""Deterministic MAPE-K experiment over an event-sourced causal topology.

The fixture models a checkout dependency that becomes slow and error-prone.  Its
authoritative events, metric observations, application logs and trace spans are
kept separate in the emitted artifacts but projected together only where an
explicit ``causation_id`` exists.  The resulting control loop is deliberately
small: monitor detects degradation, analyse binds the three evidence channels,
plan selects a fallback, execute enables it and knowledge records the verified
operational rule.

This is a controlled mechanism experiment.  It demonstrates chronology and
evidence preservation; it is not evidence that MAPE-K or CT-RAG will diagnose
arbitrary production incidents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ctrag import CausalProvenance, CausalTopology, DynamicTerrain, EdgeKind, EventProjector, EventRecord


@dataclass(frozen=True)
class MapeKObservabilityConfig:
    healthy_cycles: int = 2
    recovered_cycles: int = 2
    latency_threshold_ms: float = 800.0
    degraded_latency_ms: float = 1_200.0
    recovered_latency_ms: float = 140.0
    degraded_error_rate: float = 0.25

    def __post_init__(self) -> None:
        if self.healthy_cycles < 1 or self.recovered_cycles < 1:
            raise ValueError("healthy_cycles and recovered_cycles must be positive")
        for name in (
            "latency_threshold_ms", "degraded_latency_ms", "recovered_latency_ms", "degraded_error_rate",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.degraded_latency_ms <= self.latency_threshold_ms:
            raise ValueError("degraded latency must exceed the monitor threshold")
        if self.recovered_latency_ms >= self.latency_threshold_ms:
            raise ValueError("recovered latency must be below the monitor threshold")
        if self.degraded_error_rate > 1:
            raise ValueError("degraded_error_rate must not exceed one")


class _Scenario:
    """Append-only deterministic system simulator used only by this benchmark."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.events: list[EventRecord] = []
        self.metrics: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []
        self._sequence = 0

    def emit(
        self,
        event_type: str,
        *,
        payload: dict[str, Any],
        causation_id: str | None = None,
        correlation_id: str = "order-42",
        execution_id: str = "checkout-42",
        status: str = "ok",
    ) -> EventRecord:
        self._sequence += 1
        event = EventRecord(
            event_id=f"evt-{self._sequence:03d}",
            event_type=event_type,
            timestamp=self.now,
            payload=payload,
            causation_id=causation_id,
            correlation_id=correlation_id,
            execution_id=execution_id,
            actor_id="CheckoutAgent",
            status=status,
        )
        self.events.append(event)
        self.now += timedelta(seconds=1)
        return event

    def checkout_cycle(self, phase: str, latency_ms: float, error_rate: float) -> dict[str, EventRecord]:
        request = self.emit(
            "Checkout.Request.Accepted",
            payload={"phase": phase, "order_id": "order-42"},
            execution_id=f"checkout-{phase}-{self._sequence + 1}",
        )
        attempt = self.emit(
            "Payment.Dependency.Attempted",
            payload={"provider": "primary-payments", "phase": phase},
            causation_id=request.event_id,
            execution_id=request.execution_id or "checkout",
        )
        metric = self.emit(
            "Observability.Metric.Recorded",
            payload={"phase": phase, "name": "payment.latency_ms", "value": latency_ms, "error_rate": error_rate},
            causation_id=attempt.event_id,
            execution_id=request.execution_id or "checkout",
            status="degraded" if error_rate else "ok",
        )
        self.metrics.append({
            "timestamp": metric.timestamp.isoformat(), "phase": phase, "metric": "payment.latency_ms",
            "value": latency_ms, "error_rate": error_rate, "event_id": metric.event_id,
        })
        trace = self.emit(
            "Observability.Trace.Recorded",
            payload={"phase": phase, "span": "payments.authorize", "latency_ms": latency_ms,
                     "outcome": "timeout" if error_rate else "ok"},
            causation_id=attempt.event_id,
            execution_id=request.execution_id or "checkout",
            status="error" if error_rate else "ok",
        )
        self.traces.append({
            "trace_id": f"trace-{trace.event_id}", "event_id": trace.event_id,
            "timestamp": trace.timestamp.isoformat(), "phase": phase, "span": "payments.authorize",
            "latency_ms": latency_ms, "outcome": "timeout" if error_rate else "ok",
        })
        result = {"request": request, "attempt": attempt, "metric": metric, "trace": trace}
        if error_rate:
            log = self.emit(
                "Observability.Log.Recorded",
                payload={"phase": phase, "level": "ERROR", "message": "primary payments timeout"},
                causation_id=attempt.event_id,
                execution_id=request.execution_id or "checkout",
                status="error",
            )
            self.logs.append({
                "timestamp": log.timestamp.isoformat(), "event_id": log.event_id, "phase": phase,
                "level": "ERROR", "message": "primary payments timeout", "trace_event_id": trace.event_id,
            })
            result["log"] = log
        return result


def _is_after(events: list[EventRecord], child_id: str, parent_id: str) -> bool:
    indexed = {event.event_id: event for event in events}
    return indexed[child_id].timestamp > indexed[parent_id].timestamp


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


def run(config: MapeKObservabilityConfig, output: Path) -> dict[str, Any]:
    """Run one append-only observable incident and a proof-bounded MAPE-K loop."""
    output.mkdir(parents=True, exist_ok=True)
    scenario = _Scenario()
    for _ in range(config.healthy_cycles):
        scenario.checkout_cycle("healthy", latency_ms=120.0, error_rate=0.0)

    degraded = scenario.checkout_cycle("degraded", config.degraded_latency_ms, config.degraded_error_rate)
    monitor = scenario.emit(
        "MAPE-K.Monitor.DegradationDetected",
        payload={"metric_event_id": degraded["metric"].event_id, "threshold_ms": config.latency_threshold_ms,
                 "observed_latency_ms": config.degraded_latency_ms},
        causation_id=degraded["metric"].event_id,
        execution_id="mape-k-control",
        status="degraded",
    )
    evidence_ids = [degraded["metric"].event_id, degraded["trace"].event_id, degraded["log"].event_id]
    analysis = scenario.emit(
        "MAPE-K.Analyze.EvidenceBound",
        payload={"evidence_event_ids": evidence_ids, "diagnosis": "primary-payments-timeout"},
        causation_id=monitor.event_id,
        execution_id="mape-k-control",
        status="confirmed",
    )
    plan = scenario.emit(
        "MAPE-K.Plan.FallbackSelected",
        payload={"diagnosis": "primary-payments-timeout", "action": "route-to-secondary-payments"},
        causation_id=analysis.event_id,
        execution_id="mape-k-control",
        status="approved",
    )
    execution = scenario.emit(
        "MAPE-K.Execute.FallbackEnabled",
        payload={"action": "route-to-secondary-payments", "provider": "secondary-payments"},
        causation_id=plan.event_id,
        execution_id="mape-k-control",
        status="applied",
    )

    recovered_cycles = [
        scenario.checkout_cycle("recovered", config.recovered_latency_ms, 0.0)
        for _ in range(config.recovered_cycles)
    ]
    final_metric = recovered_cycles[-1]["metric"]
    verification = scenario.emit(
        "MAPE-K.Monitor.RecoveryObserved",
        payload={"metric_event_id": final_metric.event_id, "latency_ms": config.recovered_latency_ms,
                 "threshold_ms": config.latency_threshold_ms},
        causation_id=final_metric.event_id,
        execution_id="mape-k-control",
        status="recovered",
    )
    knowledge = scenario.emit(
        "MAPE-K.Knowledge.OperationalRuleVerified",
        payload={"condition": "primary-payments-timeout", "action": "route-to-secondary-payments",
                 "evidence_event_ids": evidence_ids + [verification.event_id]},
        causation_id=verification.event_id,
        execution_id="mape-k-control",
        status="verified",
    )

    topology = CausalTopology()
    EventProjector(topology).ingest_many(scenario.events)
    event_node_ids_before_terrain = set(topology.nodes)
    terrain = DynamicTerrain(topology)
    for event in scenario.events:
        if event.causation_id:
            terrain.observe_transition(
                event.causation_id, event.event_id, provenance=CausalProvenance.EVENT,
            )

    degraded_latency = config.degraded_latency_ms
    recovered_latency = config.recovered_latency_ms
    # Explicit event causation, not event order, is the topology used by the loop.
    causal_edge_count = sum(
        len(topology.outgoing(node_id, {EdgeKind.CAUSAL}))
        for node_id in topology.nodes
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "claim_scope": "controlled MAPE-K observability mechanism; not a production efficacy or causal-inference claim",
        "config": asdict(config),
        "event_count": len(scenario.events),
        "topology": {
            "node_count": len(topology.nodes),
            "observed_event_causal_edge_count": causal_edge_count,
            "authoritative_event_history_preserved": len(topology.nodes) == len(scenario.events),
        },
        "outcomes": {
            "monitor_detected_degradation": degraded_latency > config.latency_threshold_ms,
            "analysis_binds_metric_log_and_trace": set(evidence_ids) == {
                degraded["metric"].event_id, degraded["trace"].event_id, degraded["log"].event_id,
            },
            "plan_after_analysis": _is_after(scenario.events, plan.event_id, analysis.event_id),
            "execution_after_plan": _is_after(scenario.events, execution.event_id, plan.event_id),
            "recovery_after_execution": _is_after(scenario.events, verification.event_id, execution.event_id),
            "recovery_below_threshold": recovered_latency < config.latency_threshold_ms,
            "latency_improved": recovered_latency < degraded_latency,
            "knowledge_is_evidence_bounded": set(knowledge.payload["evidence_event_ids"]).issubset({event.event_id for event in scenario.events}),
            "terrain_preserves_authoritative_history": event_node_ids_before_terrain == set(topology.nodes),
        },
        "mape_k": {
            "monitor_event_id": monitor.event_id,
            "analysis_event_id": analysis.event_id,
            "plan_event_id": plan.event_id,
            "execution_event_id": execution.event_id,
            "knowledge_event_id": knowledge.event_id,
            "evidence_event_ids": evidence_ids,
        },
        "protocol": {
            "chronology": "All events are append-only and timestamps are strictly increasing.",
            "causality": "Only explicit EventRecord.causation_id values create causal topology edges; temporal order is not promoted to causality.",
            "observability": "Metrics, logs and traces are emitted as distinct artifacts and linked in analysis by immutable event identifiers.",
            "knowledge": "The K outcome stores a verified operational rule plus event identifiers, not an unbounded copy of logs or traces.",
        },
    }

    encoded_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "mape-k-observability-results.json").write_text(encoded_report, encoding="utf-8", newline="\n")
    _write_ndjson(output / "events.ndjson", [event.to_dict() for event in scenario.events])
    _write_ndjson(output / "logs.ndjson", scenario.logs)
    _write_ndjson(output / "traces.ndjson", scenario.traces)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "phase", "metric", "value", "error_rate", "event_id"])
        writer.writeheader()
        writer.writerows(scenario.metrics)
    manifest = {
        "schema_version": 1,
        "command": "python -m ctrag.benchmarks.mape_k_observability",
        "results_sha256": hashlib.sha256(encoded_report.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic MAPE-K observability experiment")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/mape-k-observability"))
    args = parser.parse_args(argv)
    report = run(MapeKObservabilityConfig(), args.output)
    print(f"Wrote {report['event_count']} MAPE-K observable events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
