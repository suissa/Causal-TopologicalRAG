import hashlib
import inspect
import json

import pytest

from ctrag.experiments.commercial_system import (
    SCENARIOS,
    CommercialExperimentConfig,
    assert_no_oracle_leakage,
    causal_frontier_diagnosis,
    evaluate,
    generate,
    project,
    run,
)
from ctrag.models import CausalProvenance, EdgeKind


def test_generator_is_deterministic_and_oracle_is_separate():
    config = CommercialExperimentConfig()
    observations, oracle = generate(config)
    repeated, repeated_oracle = generate(config)

    assert [event.to_dict() for event in observations] == [event.to_dict() for event in repeated]
    assert oracle == repeated_oracle
    assert len(oracle) == len(SCENARIOS) == 7
    assert len({case.root_cause_event_id for case in oracle}) == 7
    assert all(case.root_cause_event_id.startswith("evt-") for case in oracle)
    assert all("cause" not in case.root_cause_event_id for case in oracle)


def test_projection_preserves_explicit_causation_and_blocks_oracle_fields():
    observations, oracle = generate(CommercialExperimentConfig())
    topology = project(observations)
    assert_no_oracle_leakage(topology)

    for case in oracle:
        path = topology.causal_path_evidence(
            case.symptom_event_id,
            case.root_cause_event_id,
            direction="in",
            max_hops=8,
        )
        assert path is not None
        assert path.nodes[0] == case.symptom_event_id
        assert path.nodes[-1] == case.root_cause_event_id
        assert all(edge.kind is EdgeKind.CAUSAL for edge in path.edges)


def test_decoys_are_similar_but_not_causal_ancestors():
    observations, oracle = generate(CommercialExperimentConfig())
    topology = project(observations)
    for case in oracle:
        decoys = [
            node for node in topology.nodes.values()
            if node.metadata.get("correlation_id") in {
                f"knowledge:{case.scenario_id}", f"training:{case.scenario_id}"
            }
        ]
        assert len(decoys) == 2
        assert all(
            topology.causal_path_evidence(
                case.symptom_event_id, node.id, direction="in", max_hops=8
            ) is None
            for node in decoys
        )


def test_frontier_api_cannot_receive_oracle_answer():
    parameters = inspect.signature(causal_frontier_diagnosis).parameters
    assert "root_cause_event_id" not in parameters
    assert "oracle" not in parameters


def test_inferred_scenario_has_no_parent_causation_for_signals():
    observations, oracle = generate(CommercialExperimentConfig())
    case = next(item for item in oracle if item.scenario_id == "pricing_configuration_drift")
    signals = [event for event in observations if event.correlation_id == f"incident:{case.scenario_id}"]
    independent = {
        "Config.PricingRuleVersionDrift", "Metrics.OrderConversionDrop",
        "Trace.CheckoutDiscountMismatch", "Logs.PromotionVersionMismatch",
        "Sales.OrderConversionDropped",
    }
    assert all(event.causation_id is None for event in signals if event.event_type in independent)

    topology = project(observations)
    inferred = [
        edge for edge in topology.outgoing(case.root_cause_event_id, {EdgeKind.CAUSAL})
        if edge.provenance is CausalProvenance.INFERRED
    ]
    assert len(inferred) >= 3
    assert all(edge.confidence < 1.0 for edge in inferred)

    result = evaluate(topology, oracle, CommercialExperimentConfig())
    frontier = next(
        row for row in result["rows"]
        if row["scenario_id"] == case.scenario_id
        and row["task"] == "diagnosis"
        and row["arm"] == "ctrag_causal_frontier"
    )
    assert frontier["top1"] == 1


def test_all_known_causes_and_solutions_are_recovered():
    config = CommercialExperimentConfig()
    observations, oracle = generate(config)
    result = evaluate(project(observations), oracle, config)

    frontier = next(row for row in result["summary"] if row["arm"] == "ctrag_causal_frontier")
    recovery = next(
        row for row in result["summary"]
        if row["task"] == "recovery" and row["arm"] == "full_ctrag"
    )
    lexical = next(
        row for row in result["summary"]
        if row["task"] == "diagnosis" and row["arm"] == "lexical_only"
    )
    assert frontier["top1_accuracy"] == 1.0
    assert recovery["recall_at_3"] == 1.0
    assert lexical["top1_accuracy"] < frontier["top1_accuracy"]


def test_artifacts_are_reproducible_and_explorer_defaults_to_blind(tmp_path):
    result = run(tmp_path)
    expected = {
        "REPORT.md", "explorer.html", "graph.json", "manifest.json",
        "observations.ndjson", "oracle.json", "results.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert result == json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        content = (tmp_path / name).read_bytes()
        assert record["bytes"] == len(content)
        assert record["sha256"] == hashlib.sha256(content).hexdigest()

    explorer = (tmp_path / "explorer.html").read_text(encoding="utf-8")
    assert 'id="reveal" type="checkbox"' in explorer
    assert 'id="reveal" type="checkbox" checked' not in explorer
    assert "https://" not in explorer


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"dimensions": 0}, {"max_hops": 0}, {"k": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        CommercialExperimentConfig(**kwargs)
