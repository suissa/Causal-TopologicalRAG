from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    Edge,
    EdgeKind,
    MemoryNode,
    QueryMode,
    RetrievalWeights,
)
from ctrag.benchmarks.datasets import generate
from ctrag.benchmarks.metrics import evaluate, ranking_metrics


def _direction_fixture() -> CausalTopology:
    topology = CausalTopology()
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    for offset, (node_id, text) in enumerate((
        ("ancestor", "historical precursor"),
        ("anchor", "current failure"),
        ("recovered", "successful recovery"),
        ("future", "later consequence"),
    )):
        topology.add_node(MemoryNode(node_id, text, now + timedelta(seconds=offset)))
    for source, target in (("ancestor", "anchor"), ("anchor", "recovered"), ("recovered", "future")):
        topology.add_edge(Edge(
            source,
            target,
            EdgeKind.CAUSAL,
            provenance=CausalProvenance.EXECUTION,
        ))
    return topology


def _causal_only() -> RetrievalWeights:
    return RetrievalWeights(0, 0, 1, 0, 0, 0)


def test_recovery_causal_signal_is_descendant_only_regression() -> None:
    """Regression: RECOVERY used to score both ancestors and descendants as causal."""
    hits = CTRetriever(_direction_fixture()).search(
        "recover current failure",
        mode=QueryMode.RECOVERY,
        anchor_ids=["anchor"],
        weights=_causal_only(),
        exhaustive=True,
        k=4,
    )
    by_id = {hit.node.id: hit for hit in hits}
    assert by_id["recovered"].components["causal"] > 0
    assert by_id["future"].components["causal"] > 0
    assert by_id["ancestor"].components["causal"] == 0


def test_counterfactual_causal_signal_is_ancestor_only_regression() -> None:
    """Observational divergence support must not promote consequences as causes."""
    hits = CTRetriever(_direction_fixture()).search(
        "where could this trajectory have diverged",
        mode=QueryMode.COUNTERFACTUAL,
        anchor_ids=["anchor"],
        weights=_causal_only(),
        exhaustive=True,
        k=4,
    )
    by_id = {hit.node.id: hit for hit in hits}
    assert by_id["ancestor"].components["causal"] > 0
    assert by_id["recovered"].components["causal"] == 0
    assert by_id["future"].components["causal"] == 0


def test_adversarial_semantic_lookalike_is_not_causal_evidence() -> None:
    topology = CausalTopology()
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    topology.add_node(MemoryNode("cause", "lease expired before reservation", now))
    topology.add_node(MemoryNode("anchor", "inventory reservation failed", now + timedelta(seconds=1)))
    topology.add_node(MemoryNode("lookalike", "inventory reservation failed troubleshooting report", now + timedelta(seconds=2)))
    topology.add_edge(Edge("cause", "anchor", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION))
    topology.add_edge(Edge("lookalike", "anchor", EdgeKind.TEMPORAL))

    hits = CTRetriever(topology).search(
        "inventory reservation failed",
        mode=QueryMode.WHY,
        anchor_ids=["anchor"],
        weights=_causal_only(),
        exhaustive=True,
        k=2,
    )
    by_id = {hit.node.id: hit for hit in hits}
    assert by_id["cause"].components["causal"] > 0
    assert by_id["lookalike"].components["causal"] == 0


def test_adversarial_temporal_chain_cannot_satisfy_causal_path() -> None:
    dataset = generate("failure_recovery", 7, 1)
    query = dataset.queries[0]
    causal_path = query.causal_paths[0]

    temporal = CausalTopology()
    for node in dataset.topology.nodes.values():
        temporal.add_node(node)
    for source, target in zip(causal_path, causal_path[1:]):
        temporal.add_edge(Edge(source, target, EdgeKind.TEMPORAL))

    ranked = [node_id for node_id in causal_path if node_id != query.anchor]
    metrics = evaluate(ranked, query, temporal, len(ranked))
    assert metrics["causal_path_recall"] == 0
    assert metrics["causal_distance_error"] > 0


def test_adversarial_wrong_direction_does_not_count_as_complete_path() -> None:
    topology = _direction_fixture()
    # Gold says ancestor -> anchor, but retrieved context has only descendants.
    from ctrag.benchmarks.datasets import Query

    query = Query(
        id="why-direction",
        text="why current failure",
        mode=QueryMode.WHY,
        anchor="anchor",
        relevance={"ancestor": 2},
        causal_nodes=["ancestor"],
        causal_paths=[["ancestor", "anchor"]],
        causal_distances={"ancestor": 1},
    )
    metrics = evaluate(["recovered", "future"], query, topology, 2)
    assert metrics["causal_recall_at_k"] == 0
    assert metrics["causal_path_recall"] == 0
    assert metrics["causal_distance_error"] == len(topology.nodes)


@pytest.mark.parametrize("name", ["failure_recovery", "branching"])
def test_synthetic_benchmark_has_no_id_or_label_text_leakage(name: str) -> None:
    dataset = generate(name, 7, 4)
    for query in dataset.queries:
        # Opaque node IDs and gold relevant IDs must not be exposed in natural-language queries.
        assert query.anchor not in query.text
        assert all(node_id not in query.text for node_id in query.relevance)
        # Oracle anchor is evaluation configuration, never itself a relevant answer.
        assert query.anchor not in query.relevance
        assert query.anchor not in query.causal_nodes
        # Ground-truth IDs must correspond to actual nodes, not hidden synthetic objects.
        assert set(query.relevance) <= set(dataset.topology.nodes)
        assert set(query.causal_nodes) <= set(dataset.topology.nodes)


def test_metric_contract_hand_check_for_empty_and_partial_rankings() -> None:
    empty = ranking_metrics([], {"a": 2, "b": 1}, 3)
    assert empty == {
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }
    partial = ranking_metrics(["noise", "a"], {"a": 2, "b": 1}, 2)
    assert partial["recall_at_k"] == 0.5
    assert partial["precision_at_k"] == 0.5
    assert partial["mrr"] == 0.5
    assert 0 < partial["ndcg"] < 1
