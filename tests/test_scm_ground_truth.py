import copy

import pytest

from ctrag.benchmarks.scm_ground_truth import (
    FAMILIES, assert_no_label_leakage, build_topology, evaluate_family,
    generate_records, public_view, run,
)
from ctrag.models import MemoryNode


@pytest.mark.parametrize("family", FAMILIES)
def test_scm_has_exact_paired_ground_truth_and_fixed_effect(family):
    records = generate_records(family, 101, n=30)
    assert all(record.true_individual_effect == pytest.approx(2.0) for record in records)
    assert all(record.counterfactual_y != record.factual_y for record in records)


def test_public_index_view_excludes_counterfactual_and_exogenous_answers():
    record = generate_records("hidden_confounder", 101, n=3)[0]
    view = public_view(record)
    assert "counterfactual_y" not in view
    assert "true_individual_effect" not in view
    assert view["z"] is None


def test_leakage_guard_rejects_answer_fields():
    record = generate_records("linear_chain", 101, n=3)[0]
    topology = build_topology([record], include_true_causal_edges=False)
    assert_no_label_leakage(topology.nodes[record.id])
    leaky = copy.deepcopy(topology.nodes[record.id])
    leaky.metadata["counterfactual_y"] = 99
    with pytest.raises(RuntimeError, match="answer-label leakage"):
        assert_no_label_leakage(leaky)


def test_supplied_graph_retrieval_is_not_reported_as_discovery():
    result = evaluate_family("observed_confounder", 101)
    assert result["causal_discovery"]["performed"] is False
    assert result["counterfactual_estimation"]["coverage"] == 0.0
    assert result["intervention_pair_recall_at_2"]["ctrag_supplied_graph"] > result["intervention_pair_recall_at_2"]["dense"]


def test_artifact_is_machine_readable_and_holdout_stays_sealed(tmp_path):
    result = run(tmp_path)
    assert len(result["results"]) == 15
    assert result["final_holdout"] == "sealed_not_loaded"
    assert (tmp_path/"scm-results.json").is_file()
    assert (tmp_path/"manifest.json").is_file()
