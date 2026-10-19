from datetime import datetime, timedelta, timezone

from ctrag.experiments.codemanager import EvidenceSignal, rank_hypotheses


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def signal(event_id, seconds, kind, status, **kwargs):
    return EvidenceSignal(
        event_id=event_id,
        timestamp=T0 + timedelta(seconds=seconds),
        kind=kind,
        status=status,
        evidence_group=kwargs.pop("evidence_group", "incident-a"),
        **kwargs,
    )


def test_crosses_code_artifact_with_distinct_independent_signals():
    rows = [
        signal("release", 0, "code", "error", artifact="price-serializer",
               consumer_artifact="checkout-api", expected_contract="integer-cents",
               loaded_contract="float"),
        signal("trace", 1, "trace", "warning", artifact="checkout-api",
               uses_artifact="price-serializer", independent_source="trace-backend"),
        signal("log", 2, "log", "error", artifact="checkout-api",
               uses_artifact="price-serializer", independent_source="log-backend"),
        signal("metric", 3, "metric", "warning", consumer_artifact="checkout-api",
               attributes={"uses": "price-serializer"}, independent_source="metrics"),
    ]
    ranked = rank_hypotheses(rows)
    assert ranked[0].event_id == "release"
    assert set(ranked[0].evidence) == {"trace", "log", "metric"}
    assert ranked[0].score <= 0.99
    assert ranked[0].score > 0.7


def test_duplicate_samples_from_same_source_do_not_inflate_support():
    candidate = signal("config", 0, "config", "warning", artifact="worker",
                       expected_version="v2", loaded_version="v1")
    duplicate_a = signal("log-1", 1, "log", "error", artifact="worker",
                         independent_source="same-log-index")
    duplicate_b = signal("log-2", 2, "log", "error", artifact="worker",
                         independent_source="same-log-index")
    one = rank_hypotheses([candidate, duplicate_a])
    two = rank_hypotheses([candidate, duplicate_a, duplicate_b])
    assert one[0].score == two[0].score
    assert one[0].evidence == two[0].evidence


def test_unrelated_group_and_non_linked_artifact_are_not_joined():
    candidate = signal("config", 0, "config", "warning", artifact="worker")
    unrelated_group = signal("trace", 1, "trace", "error", artifact="worker",
                             evidence_group="incident-b")
    unrelated_artifact = signal("log", 2, "log", "error", artifact="other-service")
    ranked = rank_hypotheses([candidate, unrelated_group, unrelated_artifact])
    assert ranked[0].evidence == ()
    assert ranked[0].score < 0.6


def test_counter_evidence_reduces_score_and_is_reported():
    candidate = signal("code", 0, "code", "error", artifact="serializer")
    anomalous = signal("bad-trace", 1, "trace", "warning", artifact="serializer")
    healthy = signal("healthy-trace", 2, "trace", "healthy", artifact="serializer",
                     independent_source="healthy-canary")
    without_counter = rank_hypotheses([candidate, anomalous])[0]
    with_counter = rank_hypotheses([candidate, anomalous, healthy])[0]
    assert with_counter.score < without_counter.score
    assert with_counter.counter_evidence == ("healthy-trace",)


def test_code_and_config_contract_mismatch_are_both_candidates():
    rows = [
        signal("code", 0, "code", "error", artifact="tax-client",
               consumer_artifact="pricing-worker", expected_version="v4", loaded_version="v3"),
        signal("config", 0, "config", "warning", artifact="pricing-worker",
               uses_artifact="tax-client", expected_version="v4", loaded_version="v3"),
        signal("trace", 1, "trace", "warning", artifact="pricing-worker",
               uses_artifact="tax-client"),
    ]
    ranked = rank_hypotheses(rows)
    assert {item.event_id for item in ranked} == {"code", "config"}
    assert all(item.score < 0.99 for item in ranked)



def test_leave_one_independent_signal_out_reduces_evidence_support():
    candidate = signal("release", 0, "code", "error", artifact="serializer")
    trace = signal("trace", 1, "trace", "warning", artifact="serializer",
                   independent_source="trace-backend")
    log = signal("log", 2, "log", "error", artifact="serializer",
                 independent_source="log-backend")
    metric = signal("metric", 3, "metric", "warning", artifact="serializer",
                    independent_source="metrics")
    full = rank_hypotheses([candidate, trace, log, metric])[0]
    without_trace = rank_hypotheses([candidate, log, metric])[0]
    assert full.score > without_trace.score
    assert set(full.evidence) == {"trace", "log", "metric"}
    assert set(without_trace.evidence) == {"log", "metric"}


def test_candidate_does_not_join_future_evidence_from_another_group():
    candidate = signal("release", 0, "code", "error", artifact="serializer")
    later_unrelated = signal("trace", 1, "trace", "error", artifact="serializer",
                             evidence_group="other-incident")
    ranked = rank_hypotheses([candidate, later_unrelated])
    assert ranked[0].evidence == ()
