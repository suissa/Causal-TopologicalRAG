from __future__ import annotations

import copy
from datetime import timedelta

import pytest

from ctrag import CausalClaimError, CausalTopology, EvidenceLevel, MemoryNode, validate_causal_claim
from ctrag.benchmarks.datasets import generate
from ctrag.benchmarks.gate_d import (
    FutureLeakageError,
    assert_no_future_visibility,
    chronological_snapshot,
    corrupt_causal,
    destroy_topology,
    evidence_contract_artifact,
    prospective_evaluation,
    robustness_curves,
    topology_placebos,
)
from ctrag.models import EdgeKind


def edge_identities(topology):
    return {edge.identity() for node_id in topology.nodes for edge in topology.outgoing(node_id)}


def degrees(topology):
    return sorted((len(topology.incoming(node)), len(topology.outgoing(node))) for node in topology.nodes)


def test_degree_preserving_placebo_keeps_corpus_and_degree_distribution():
    dataset = generate("failure_recovery", 7)
    placebo = destroy_topology(dataset, "degree_preserving_permutation", 99)
    assert set(placebo.nodes) == set(dataset.topology.nodes)
    assert [placebo.nodes[n].text for n in sorted(placebo.nodes)] == [dataset.topology.nodes[n].text for n in sorted(dataset.topology.nodes)]
    assert degrees(placebo) == degrees(dataset.topology)
    assert edge_identities(placebo) != edge_identities(dataset.topology)


@pytest.mark.parametrize("control", ["remove_causal", "remove_temporal", "random_direction", "causation_permutation", "topology_only_sham"])
def test_destruction_controls_preserve_node_corpus(control):
    dataset = generate("branching", 42)
    placebo = destroy_topology(dataset, control, 123)
    assert set(placebo.nodes) == set(dataset.topology.nodes)
    assert {n: placebo.nodes[n].text for n in placebo.nodes} == {n: dataset.topology.nodes[n].text for n in dataset.topology.nodes}


def test_corruption_levels_form_machine_readable_curves_by_query_mode():
    rows = robustness_curves((7,))
    assert {row["level"] for row in rows} == {0.0, .25, .5, .75, 1.0}
    assert {row["mode"] for row in rows} >= {"why", "what_next", "recovery"}
    assert {row["provenance_weighted"] for row in rows} == {True, False}
    assert {row["corruption"] for row in rows} >= {"missing", "noisy", "incorrect_direction"}


def test_full_causal_removal_degrades_causal_path_recall():
    rows = topology_placebos((7,))
    intact = [r for r in rows if r["control"] == "intact"]
    removed = [r for r in rows if r["control"] == "remove_causal"]
    assert sum(r["causal_evidence_rate"] for r in removed) < sum(r["causal_evidence_rate"] for r in intact)


def test_observational_contract_rejects_counterfactual_headline_claim():
    validate_causal_claim("historical divergence provides observational support", EvidenceLevel.OBSERVATIONAL_SUPPORT)
    with pytest.raises(CausalClaimError):
        validate_causal_claim("The logs proved causality", EvidenceLevel.OBSERVATIONAL_SUPPORT)
    contract = evidence_contract_artifact()
    assert contract["confounded_negative_control"]["interventional_claim_permitted"] is False
    assert contract["interventional_evaluation"]["reported_separately"] is True


def test_chronological_snapshot_excludes_future_nodes_and_edges():
    dataset = generate("failure_recovery", 7, traces=1)
    query = next(q for q in dataset.queries if q.mode.value == "recovery")
    cutoff = dataset.topology.nodes[query.anchor].timestamp
    snapshot = chronological_snapshot(dataset, cutoff)
    assert snapshot.nodes
    assert all(node.timestamp <= cutoff for node in snapshot.nodes.values())
    assert all(e.source in snapshot.nodes and e.target in snapshot.nodes for node in snapshot.nodes for e in snapshot.outgoing(node))


def test_anti_leakage_check_fails_on_future_node():
    dataset = generate("failure_recovery", 7, traces=1)
    cutoff = min(node.timestamp for node in dataset.topology.nodes.values())
    leaky = CausalTopology()
    future = max(dataset.topology.nodes.values(), key=lambda node: node.timestamp)
    leaky.add_node(copy.deepcopy(future))
    with pytest.raises(FutureLeakageError):
        assert_no_future_visibility(leaky, cutoff)


def test_prospective_manifest_records_cutoff_and_exclusions():
    rows = prospective_evaluation((7,))
    assert rows
    assert all(row["evaluation"] == "prospective" for row in rows)
    assert all(row["cutoff"].endswith("+00:00") for row in rows)
    assert all(row["future_nodes_excluded"] > 0 for row in rows)


def test_corruption_does_not_mutate_original():
    dataset = generate("failure_recovery", 7)
    before = edge_identities(dataset.topology)
    corrupt_causal(dataset, "removed", .5, 4)
    assert edge_identities(dataset.topology) == before
