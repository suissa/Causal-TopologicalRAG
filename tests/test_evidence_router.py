from __future__ import annotations

import pytest

from ctrag import (
    ConfigTreeAdapter,
    EvidenceRoute,
    EvidenceRouter,
    EvidenceShape,
    LocalRelationKind,
    MetricSeriesAdapter,
    RawEvidence,
    TraceAdapter,
)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("settings.yaml", "database:\n  host: localhost\n"),
        ("core.toml", '[database]\nhost = "localhost"\n'),
        ("app-config.json", '{"database":{"host":"localhost"}}'),
    ],
)
def test_config_formats_route_to_config_tree(filename: str, content: str) -> None:
    route = EvidenceRouter().route(RawEvidence("cfg", content, filename=filename))
    assert route.shape is EvidenceShape.CONFIG_TREE
    assert route.adapter_id == "config-tree-v1"
    assert "path" in route.preserve
    assert "value" in route.preserve


def test_otel_like_span_routes_to_trace() -> None:
    evidence = RawEvidence(
        "trace",
        {
            "trace_id": "t1",
            "span_id": "s1",
            "parent_span_id": "s0",
            "name": "db.query",
            "attributes": {"db.system": "postgres"},
        },
    )
    route = EvidenceRouter().route(evidence)
    assert route.shape is EvidenceShape.TRACE
    assert route.adapter_id == "trace-v1"


def test_labeled_series_routes_to_metric_series() -> None:
    evidence = RawEvidence(
        "metric",
        {
            "metric": "latency_ms",
            "labels": {"service": "checkout"},
            "unit": "ms",
            "samples": [
                {"timestamp": "2026-09-20T00:00:00Z", "value": 10.0},
                {"timestamp": "2026-09-20T00:01:00Z", "value": 12.0},
            ],
        },
    )
    route = EvidenceRouter().route(evidence)
    assert route.shape is EvidenceShape.METRIC_SERIES
    assert route.adapter_id == "metric-series-v1"


def test_ambiguous_payload_is_not_silently_coerced() -> None:
    route = EvidenceRouter().route(RawEvidence("ambiguous", {"foo": "bar"}))
    assert route.shape is None
    assert route.fallback_required is True
    assert route.method_id == "deterministic:undetermined"


class _Fallback:
    def classify(self, evidence: RawEvidence) -> EvidenceRoute | None:
        return EvidenceRoute(
            shape=EvidenceShape.STRUCTURED_PAYLOAD,
            confidence=0.61,
            method_id="semantic:test-v1",
            adapter_id=None,
            preserve=("content",),
            source_metadata={"source_id": evidence.source_id},
            classifier_version="semantic-test-v1",
        )


def test_semantic_fallback_runs_only_after_deterministic_router_is_undetermined() -> None:
    router = EvidenceRouter(semantic_fallback=_Fallback())

    ambiguous = router.route(RawEvidence("ambiguous", {"foo": "bar"}))
    assert ambiguous.shape is EvidenceShape.STRUCTURED_PAYLOAD
    assert ambiguous.method_id == "semantic:test-v1"

    trace = router.route(RawEvidence("trace", {"trace_id": "t", "span_id": "s"}))
    assert trace.shape is EvidenceShape.TRACE
    assert trace.method_id == "schema:trace"


def test_config_adapter_preserves_resolution_chain_not_opaque_effective_value() -> None:
    chain = [
        {"source": "default", "value": 10, "precedence": 0, "winner": False},
        {"source": "env", "value": 20, "precedence": 10, "winner": True},
    ]
    evidence = RawEvidence(
        "cfg",
        {"database": {"pool_size": 20}},
        filename="core.toml",
        metadata={"resolution_chain": {"database.pool_size": chain}},
    )
    route = EvidenceRouter().route(evidence)
    anchors = ConfigTreeAdapter().adapt(evidence, route)
    anchor = next(item for item in anchors if item.structural_locator["path"] == "database.pool_size")

    assert anchor.preserved_fields["value"] == 20
    assert anchor.preserved_fields["resolution_chain"] == chain
    assert "effective_value" not in anchor.preserved_fields
    assert anchor.provenance["classifier_version"] == route.classifier_version
    assert anchor.provenance["adapter_version"] == "v1"


def test_trace_parent_child_remains_artifact_local_and_never_causal() -> None:
    evidence = RawEvidence(
        "trace",
        {
            "spans": [
                {"trace_id": "t1", "span_id": "root", "name": "request"},
                {
                    "trace_id": "t1",
                    "span_id": "child",
                    "parent_span_id": "root",
                    "name": "db.query",
                },
            ]
        },
    )
    route = EvidenceRouter().route(evidence)
    anchors = TraceAdapter().adapt(evidence, route)
    child = next(item for item in anchors if item.structural_locator["span_id"] == "child")

    assert len(child.local_relations) == 1
    relation = child.local_relations[0]
    assert relation.kind is LocalRelationKind.TRACE_PARENT
    assert relation.metadata["causal"] is False
    assert all(item.kind.value != "causal" for anchor in anchors for item in anchor.local_relations)


def test_metric_adapter_preserves_series_fields_and_local_adjacency() -> None:
    evidence = RawEvidence(
        "metric",
        {
            "metric": "cpu",
            "labels": {"host": "a"},
            "unit": "percent",
            "samples": [
                {"timestamp": 1, "value": 70.0},
                {"timestamp": 2, "value": 80.0},
            ],
        },
    )
    route = EvidenceRouter().route(evidence)
    anchors = MetricSeriesAdapter().adapt(evidence, route)

    assert anchors[0].preserved_fields["metric"] == "cpu"
    assert anchors[1].preserved_fields["labels"] == {"host": "a"}
    assert anchors[1].local_relations[0].kind is LocalRelationKind.SERIES_ADJACENT
    assert anchors[1].local_relations[0].metadata["causal"] is False
