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


def test_pre_candidate_evidence_cannot_support_a_later_candidate():
    early = signal(
        "early-trace", 0, "trace", "error", artifact="serializer",
        independent_source="trace-backend",
    )
    candidate = signal("release", 10, "code", "error", artifact="serializer")
    ranked = rank_hypotheses([early, candidate])
    assert len(ranked) == 1
    assert ranked[0].event_id == "release"
    assert ranked[0].evidence == ()
    assert ranked[0].score == 0.35


def test_same_independent_source_is_deduplicated_across_signal_kinds():
    candidate = signal("release", 0, "code", "error", artifact="serializer")
    trace = signal(
        "trace", 1, "trace", "warning", artifact="serializer",
        independent_source="observability-stack",
    )
    log = signal(
        "log", 2, "log", "error", artifact="serializer",
        independent_source="observability-stack",
    )
    result = rank_hypotheses([candidate, trace, log])[0]
    assert len(result.evidence) == 1
    assert len(result.evidence_kinds) == 1


def test_score_is_bounded_with_many_independent_support_and_counter_sources():
    candidate = signal("release", 0, "config", "warning", artifact="worker")
    signals = []
    for index in range(20):
        signals.append(signal(
            f"support-{index}", index + 1, "trace", "error",
            artifact="worker", independent_source=f"trace-{index}",
        ))
        signals.append(signal(
            f"healthy-{index}", index + 30, "metric", "healthy",
            artifact="worker", independent_source=f"canary-{index}",
        ))
    results = rank_hypotheses([candidate, *signals])
    assert len(results) == 1
    assert 0.0 <= results[0].score <= 0.99
    assert len(results[0].evidence) == 20
    assert len(results[0].counter_evidence) == 20


def test_empty_evidence_group_does_not_join_unscoped_signals():
    candidate = EvidenceSignal(
        event_id="release", timestamp=T0, kind="code", status="error",
        evidence_group="", artifact="serializer",
    )
    evidence = EvidenceSignal(
        event_id="trace", timestamp=T0 + timedelta(seconds=1), kind="trace",
        status="error", evidence_group="", artifact="serializer",
        independent_source="trace-backend",
    )
    result = rank_hypotheses([candidate, evidence])[0]
    assert result.evidence == ()
