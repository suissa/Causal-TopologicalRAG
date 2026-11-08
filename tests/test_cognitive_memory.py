from datetime import datetime, timedelta, timezone

import pytest

from ctrag.cognitive_memory import (
    BehaviorSnapshot, CognitiveEdge, CognitiveEmotionalHypothesis,
    CognitiveNode, CognitiveNodeKind, CognitiveRelation, EvidenceRef,
    SQLiteCognitiveMemoryStore,
)

UTC = timezone.utc
NOW = datetime(2026, 11, 8, 12, tzinfo=UTC)


def evidence(identifier, polarity):
    return EvidenceRef(identifier, f"cteg://evidence/{identifier}", "whatsapp_signal", NOW, polarity)


def hypothesis(**overrides):
    values = dict(
        id="hyp-1", subject_id="subject-1", statement="possible elevated cognitive load",
        confidence=0.63, supporting_evidence=(evidence("ev-1", "supports"),),
        counter_evidence=(evidence("ev-2", "contradicts"),), valid_from=NOW,
        valid_until=NOW + timedelta(hours=2), source_signal_refs=("signal-1",),
        transition_refs=("transition-1",), pattern_refs=("pattern-1",),
        inference_method="behaviorid-lt-v1", model_version="model-1", skill_version="skill-1",
        created_at=NOW,
    )
    values.update(overrides)
    return CognitiveEmotionalHypothesis(**values)


def test_hypothesis_persists_with_support_counterevidence_and_versions(tmp_path):
    store = SQLiteCognitiveMemoryStore(tmp_path / "cognitive.sqlite")
    item = hypothesis()
    store.put_hypothesis(item)
    loaded = store.get_hypotheses("subject-1")[0]
    assert loaded == item
    support, counter = store.get_hypothesis_evidence("hyp-1")
    assert [ref.id for ref in support] == ["ev-1"]
    assert [ref.id for ref in counter] == ["ev-2"]
    assert loaded.inference_method == "behaviorid-lt-v1"
    assert loaded.model_version == "model-1"
    assert loaded.skill_version == "skill-1"


def test_temporal_validity_filters_hypotheses(tmp_path):
    store = SQLiteCognitiveMemoryStore(tmp_path / "cognitive.sqlite")
    store.put_hypothesis(hypothesis())
    assert len(store.get_hypotheses("subject-1", valid_at=NOW + timedelta(minutes=1))) == 1
    assert store.get_hypotheses("subject-1", valid_at=NOW + timedelta(hours=3)) == []


def test_baselines_are_scoped_to_subject(tmp_path):
    store = SQLiteCognitiveMemoryStore(tmp_path / "cognitive.sqlite")
    store.put_node(CognitiveNode("prosody-1", CognitiveNodeKind.PROSODY_BASELINE, "subject-1", {"pitch": [1, 2]}))
    store.put_node(CognitiveNode("text-1", CognitiveNodeKind.TEXT_BASELINE, "subject-1", {"lexical_rate": 0.4}))
    store.put_node(CognitiveNode("prosody-2", CognitiveNodeKind.PROSODY_BASELINE, "subject-2", {"pitch": [9]}))
    assert {node.id for node in store.get_baselines("subject-1")} == {"prosody-1", "text-1"}


def test_snapshot_round_trip_for_behaviorid_lt(tmp_path):
    store = SQLiteCognitiveMemoryStore(tmp_path / "cognitive.sqlite")
    snapshot = BehaviorSnapshot("snapshot-1", "subject-1", NOW, ("behavior-a", "behavior-b"), ("hyp-1",), ("prosody-1", "text-1"))
    store.put_snapshot(snapshot)
    assert store.get_snapshot("snapshot-1") == snapshot


def test_edges_require_existing_nodes_and_are_explicit(tmp_path):
    store = SQLiteCognitiveMemoryStore(tmp_path / "cognitive.sqlite")
    with pytest.raises(KeyError):
        store.add_edge(CognitiveEdge("observation-1", "subject-1", CognitiveRelation.OBSERVED_FOR))
    store.put_node(CognitiveNode("observation-1", CognitiveNodeKind.OBSERVATION, "subject-1"))
    store.put_node(CognitiveNode("subject-1", CognitiveNodeKind.SUBJECT, "subject-1"))
    store.add_edge(CognitiveEdge("observation-1", "subject-1", CognitiveRelation.OBSERVED_FOR))
    # Cognitive adjacency is typed, explicit, and has no automatic causal projection.


def test_invalid_hypothesis_confidence_and_evidence_polarity_are_rejected():
    with pytest.raises(ValueError):
        hypothesis(confidence=1.2)
    with pytest.raises(ValueError):
        hypothesis(supporting_evidence=(evidence("bad", "contradicts"),))


def test_hypothesis_requires_timezone_aware_validity():
    with pytest.raises(ValueError):
        hypothesis(valid_from=datetime(2026, 11, 8, 12))
