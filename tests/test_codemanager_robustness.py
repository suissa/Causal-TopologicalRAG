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


def test_input_permutation_does_not_change_rankings():
    rows = [
        signal("release", 0, "code", "error", artifact="serializer",
               expected_contract="cents-int", loaded_contract="float"),
        signal("trace", 1, "trace", "warning", artifact="serializer",
               independent_source="trace-backend"),
        signal("log", 2, "log", "error", artifact="serializer",
               independent_source="log-backend"),
        signal("healthy", 3, "trace", "healthy", artifact="serializer",
               independent_source="canary"),
    ]
    assert [x.as_dict() for x in rank_hypotheses(rows)] == [
        x.as_dict() for x in rank_hypotheses(reversed(rows))
    ]


def test_irrelevant_or_cross_incident_injection_is_invariant():
    candidate = signal("release", 0, "code", "error", artifact="serializer",
                       expected_version="v2", loaded_version="v1")
    evidence = signal("trace", 1, "trace", "warning", artifact="serializer",
                      independent_source="trace-backend")
    noise = [
        signal("other-log", 2, "log", "error", artifact="unrelated-db",
               independent_source="other-logs"),
        signal("other-group", 3, "trace", "error", artifact="serializer",
               evidence_group="incident-b", independent_source="other-traces"),
    ]
    base = rank_hypotheses([candidate, evidence])[0]
    changed = rank_hypotheses([candidate, evidence, *noise])[0]
    assert changed.as_dict() == base.as_dict()


def test_duplicate_source_does_not_inflate_support_but_counter_evidence_does_reduce_score():
    candidate = signal("config", 0, "config", "warning", artifact="worker",
                       expected_version="v18", loaded_version="v17")
    log = signal("log-1", 1, "log", "error", artifact="worker",
                 independent_source="same-logs")
    trace = signal("trace", 2, "trace", "warning", artifact="worker",
                   independent_source="checkout-traces")
    duplicate = signal("log-duplicate", 3, "log", "error", artifact="worker",
                       independent_source="same-logs")
    healthy = signal("healthy", 4, "metric", "healthy", artifact="worker",
                     independent_source="healthy-canary")
    base = rank_hypotheses([candidate, log, trace])[0]
    duplicate_run = rank_hypotheses([candidate, log, trace, duplicate])[0]
    counter_run = rank_hypotheses([candidate, log, trace, healthy])[0]
    assert duplicate_run.score == base.score
    assert duplicate_run.evidence == base.evidence
    assert counter_run.score < base.score
    assert counter_run.counter_evidence == ("healthy",)


def test_leave_one_source_out_reduces_support_and_priority():
    candidate = signal("release", 0, "code", "error", artifact="serializer")
    evidence = [
        signal("trace", 1, "trace", "warning", artifact="serializer",
               independent_source="trace-backend"),
        signal("log", 2, "log", "error", artifact="serializer",
               independent_source="log-backend"),
        signal("metric", 3, "metric", "warning", artifact="serializer",
               independent_source="metrics"),
    ]
    full = rank_hypotheses([candidate, *evidence])[0]
    ablated = [
        rank_hypotheses([candidate, *(item for item in evidence if item.event_id != removed)])[0]
        for removed in ("trace", "log", "metric")
    ]
    assert all(item.score < full.score for item in ablated)
    assert all(len(item.evidence) == 2 for item in ablated)
