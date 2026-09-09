import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ctrag import CausalTopology, Edge, EdgeKind, CausalProvenance, MemoryNode, QueryMode, CTRetriever, RetrievalWeights
from ctrag.benchmarks.datasets import Query, generate
from ctrag.benchmarks.metrics import evaluate, ranking_metrics
from ctrag.benchmarks.runner import BASELINES, Config, run


def fixture():
    topology = CausalTopology()
    for i, node in enumerate(("a", "b", "anchor", "noise")):
        topology.add_node(MemoryNode(node, "one two", datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i)))
    for a, b in (("a", "b"), ("b", "anchor")):
        topology.add_edge(Edge(a, b, EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION))
    query = Query("q", "why", QueryMode.WHY, "anchor", {"a": 1, "b": 2},
                  ["a", "b"], [["a", "b", "anchor"]], {"a": 2, "b": 1},
                  ["a", "b", "anchor"], ["a", "b", "anchor"])
    return topology, query


def test_ranking_metrics_hand_calculated():
    result = ranking_metrics(["noise", "b", "a"], {"a": 1, "b": 2}, 2)
    assert result["recall_at_k"] == .5
    assert result["precision_at_k"] == .5
    assert result["mrr"] == .5
    assert result["ndcg"] == pytest.approx((3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert ranking_metrics(["b"], {"b": 1}, 4)["precision_at_k"] == .25
    assert ranking_metrics([], {}, 2)["recall_at_k"] is None
    assert ranking_metrics([], {"a": 1}, 2)["mrr"] == 0
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a": 1}, 2)
    with pytest.raises(ValueError):
        ranking_metrics([], {}, 0)


def test_complete_and_partial_context_metrics():
    topology, query = fixture()
    result = evaluate(["b", "a"], query, topology, 2)
    for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg", "causal_recall_at_k",
                "causal_path_recall", "trajectory_reconstruction_accuracy", "basin_purity", "context_token_efficiency"):
        assert result[key] == 1
    assert result["causal_distance_error"] == 0
    assert result["recovery_path_precision"] is None
    partial = evaluate(["b", "noise"], query, topology, 2)
    assert partial["causal_distance_error"] == 2  # missing a costs |V|=4, averaged over two labels
    assert partial["causal_path_recall"] == 0
    assert partial["trajectory_reconstruction_accuracy"] == pytest.approx(2 / 3)
    assert partial["basin_purity"] == .5
    assert partial["context_token_efficiency"] == .5
    assert partial["context_tokens"] == 4
    assert evaluate(["a"], query, topology, 1)["causal_distance_error"] == 4
    assert evaluate([], query, topology, 1)["context_token_efficiency"] == 0


def test_temporal_edges_do_not_satisfy_causal_metrics():
    topology, query = fixture()
    temporal = CausalTopology()
    for node in topology.nodes.values():
        temporal.add_node(node)
    temporal.add_edge(Edge("a", "b", EdgeKind.TEMPORAL))
    temporal.add_edge(Edge("b", "anchor", EdgeKind.TEMPORAL))
    result = evaluate(["b", "a"], query, temporal, 2)
    assert result["causal_path_recall"] == 0
    assert result["causal_distance_error"] == 4


def test_recovery_direction_and_inapplicable_metrics():
    topology, query = fixture()
    recovery = replace(query, mode=QueryMode.RECOVERY, anchor="a", relevance={"b": 2, "anchor": 1},
                       causal_nodes=["b", "anchor"], causal_distances={"b": 1, "anchor": 2},
                       recovery_nodes=["b", "anchor"], trajectory=None, basin_nodes=None)
    result = evaluate(["b", "anchor"], recovery, topology, 2)
    assert result["causal_distance_error"] == 0
    assert result["recovery_path_precision"] == 1
    assert result["trajectory_reconstruction_accuracy"] is None
    assert result["basin_purity"] is None
    assert evaluate(["b", "noise"], recovery, topology, 2)["recovery_path_precision"] == .5


def test_trajectory_order_and_extraneous_nodes_are_penalized():
    topology, query = fixture()
    topology.nodes["a"].timestamp = topology.nodes["anchor"].timestamp + timedelta(seconds=1)
    assert evaluate(["b", "a"], query, topology, 2)["trajectory_reconstruction_accuracy"] == pytest.approx(2 / 3)


def test_causal_labels_can_differ_from_relevance_and_tokens_are_weighted():
    topology, query = fixture()
    topology.nodes["noise"].text = "one two three four five six"
    query = replace(query, relevance={"b": 2, "noise": 1})
    result = evaluate(["noise"], query, topology, 1)
    assert result["recall_at_k"] == .5
    assert result["causal_recall_at_k"] == 0
    query = replace(query, relevance={"b": 2})
    assert evaluate(["b", "noise"], query, topology, 2)["context_token_efficiency"] == .25


def test_branch_path_recall_requires_complete_paths():
    dataset = generate("branching", 7, 1)
    query = dataset.queries[-1]
    ranked = query.causal_paths[0][1:]
    result = evaluate(ranked, query, dataset.topology, 2)
    assert result["causal_path_recall"] == .5
    assert result["causal_recall_at_k"] == .5


def test_why_baseline_exposes_semantic_distractors():
    dataset = generate("failure_recovery", 7, 1)
    query = dataset.queries[0]
    retriever = CTRetriever(dataset.topology)
    dense = retriever.search(query.text, anchor_ids=[query.anchor], mode=query.mode,
                             weights=BASELINES["dense_only"], exhaustive=True, k=1)
    full = retriever.search(query.text, anchor_ids=[query.anchor], mode=query.mode,
                            exhaustive=True, k=1)
    assert dense[0].node.id not in query.causal_nodes
    assert full[0].node.id in query.causal_nodes


@pytest.mark.parametrize("name", ["failure_recovery", "branching"])
def test_generator_is_deterministic_and_truth_matches_explicit_edges(name):
    dataset = generate(name, 7, 2)
    assert dataset.manifest() == generate(name, 7, 2).manifest()
    assert dataset.manifest() != generate(name, 8, 2).manifest()
    assert len(dataset.topology.nodes) == 18
    assert any(q.mode is QueryMode.WHY for q in dataset.queries)
    for query in dataset.queries:
        assert query.anchor not in query.relevance
        for path in query.causal_paths:
            for a, b in zip(path, path[1:]):
                assert any(e.target == b for e in dataset.topology.outgoing(a, {EdgeKind.CAUSAL}))
        directions = "in" if query.mode is QueryMode.WHY else "out"
        distances = dataset.topology.distances(query.anchor, direction=directions, kinds={EdgeKind.CAUSAL}, max_hops=8)
        assert query.causal_distances == {n: distances[n] for n in query.causal_nodes}
    if name == "branching":
        branches = [q for q in dataset.queries if q.id.endswith("branch-next")]
        assert all(len(q.causal_paths) == 2 and q.trajectory is None for q in branches)


def test_ablation_weights_and_exhaustive_candidates():
    assert len(BASELINES) == 7
    for weights in BASELINES.values():
        if weights is not None:
            assert weights.total() == 1
    topology = CausalTopology()
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    topology.add_node(MemoryNode("anchor", "query", now))
    for i in range(30):
        topology.add_node(MemoryNode(f"noise-{i:02}", "query", now))
    topology.add_node(MemoryNode("remote", "unrelated", now, is_attractor=True))
    # Shared basin outside the 1-hop candidate neighborhood.
    topology.add_node(MemoryNode("bridge", "unrelated", now))
    for a, b in (("anchor", "bridge"), ("bridge", "remote")):
        topology.add_edge(Edge(a, b, EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION))
    retriever = CTRetriever(topology)
    graph_hits = retriever.search("query", anchor_ids=["anchor"], max_hops=1, k=40,
                                  weights=BASELINES["graph_topology_only"], exhaustive=True)
    assert len(graph_hits) == 32
    assert "remote" in [hit.node.id for hit in graph_hits[:2]]
    for baseline, weights in BASELINES.items():
        hits = retriever.search("query", anchor_ids=["anchor"], k=40, weights=weights, exhaustive=True)
        actual = weights or RetrievalWeights.for_mode(QueryMode.SIMILAR)
        for hit in hits:
            assert hit.score == pytest.approx(sum(getattr(actual, name) * score for name, score in hit.components.items()))
    lexical = retriever.search("query", anchor_ids=["anchor"], k=3, weights=BASELINES["lexical_only"], exhaustive=True)
    assert [h.node.id for h in lexical] == ["noise-00", "noise-01", "noise-02"]


def test_outputs_and_macro_aggregation(tmp_path):
    report = run(Config(seeds=(7,), ks=(1, 3), traces=1), tmp_path)
    assert len(report["results"]) == 2 * 3 * 7 * 2
    assert report == json.loads((tmp_path / "results.json").read_text())
    assert set(row["baseline"] for row in report["results"]) == set(BASELINES)
    assert report["config"]["candidate_policy"] == "exhaustive"
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(report["results"])
    assert json.loads(rows[0]["retrieved_ids"]) == report["results"][0]["retrieved_ids"]
    for row in report["summary"]:
        if row["metric"] == "recovery_path_precision" and row["mode"] == "why":
            assert row["n"] == 0 and row["mean"] is None
        if row["metric"] == "recall_at_k":
            matching = [r["recall_at_k"] for r in report["results"]
                        if all(r[key] == row[key] for key in ("dataset", "mode", "baseline", "k"))]
            assert row["mean"] == pytest.approx(sum(matching) / len(matching))


def test_cli_reproducible_across_hash_seeds(tmp_path):
    for seed in (1, 99):
        subprocess.run([sys.executable, "-m", "ctrag.benchmarks", "--seeds", "7", "--ks", "1", "3",
                        "--traces", "1", "--output", str(tmp_path / str(seed))],
                       env=dict(os.environ, PYTHONHASHSEED=str(seed)), check=True, capture_output=True)
    for file in (tmp_path / "1").iterdir():
        assert file.read_bytes() == (tmp_path / "99" / file.name).read_bytes()


@pytest.mark.parametrize("kwargs", [{"seeds": ()}, {"seeds": (7, 7)}, {"ks": (0,)}, {"ks": (1, 1)},
                                    {"traces": 0}, {"dimensions": 0}, {"max_hops": 0}, {"hop_decay": float("nan")}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
