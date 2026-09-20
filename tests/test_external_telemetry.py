from __future__ import annotations

from pathlib import Path

from ctrag.benchmarks.external_telemetry import capability_matrix, load_trace_csv
from ctrag.models import EdgeKind


def test_external_trace_adapter_preserves_structure_without_causal_promotion(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    path.write_text(
        "timestamp,trace_id,span_id,parent_span_id,service,operation,status,duration_ms\n"
        "2026-01-01T00:00:00Z,t1,s1,,checkout,request,ok,10\n"
        "2026-01-01T00:00:01Z,t1,s2,s1,payment,authorize,ok,120\n"
        "2026-01-01T00:00:02Z,t1,s3,s2,inventory,reserve,error,4800\n",
        encoding="utf-8",
    )

    topology = load_trace_csv(path, source_name="fixture")

    assert len(topology.nodes) == 3
    assert len(topology.outgoing("trace:t1:s1", {EdgeKind.TEMPORAL})) == 1
    assert len(topology.outgoing("trace:t1:s1", {EdgeKind.BEHAVIORAL})) == 1
    assert len(topology.outgoing("trace:t1:s1", {EdgeKind.CAUSAL})) == 0
    assert topology.nodes["trace:t1:s2"].metadata["evidence_shape"] == "TRACE"
    assert topology.nodes["trace:t1:s2"].metadata["causality"] == "unknown"


def test_external_dataset_capabilities_do_not_overclaim_causality() -> None:
    matrix = capability_matrix()
    assert matrix["AIOps2020"]["supports_execution_topology"] is True
    assert matrix["AIOps2020"]["supports_causal_ground_truth"] is False
    assert matrix["SMD"]["supports_execution_topology"] is False
    assert matrix["SMD"]["supports_causal_ground_truth"] is False