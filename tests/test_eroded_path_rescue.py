from __future__ import annotations

from ctrag import CTRetriever, DynamicTerrain, QueryMode, TerrainAwareRetriever, TerrainConfig
from ctrag.benchmarks.eroded_path_rescue import (
    ErodedPathConfig,
    _build,
    _edge,
    _observe_branch,
    run,
)


def test_raw_success_baseline_is_fooled_by_unreachable_high_rate_path(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    rankings = report["rankings"]
    outcomes = report["outcomes"]
    metrics = report["metrics"]

    assert outcomes["raw_success_prefers_unreachable_path"] is True
    assert rankings["raw_success_global"][0] == "global_recovered"
    assert outcomes["global_high_success_is_unreachable_from_retry"] is True
    assert outcomes["ctrag_excludes_unreachable_high_success_path"] is True
    assert metrics["raw_success_precision_at_1"] == 0.0


def test_ctrag_prefers_reachable_high_quality_eroded_recovery(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    rankings = report["rankings"]
    metrics = report["metrics"]

    assert report["outcomes"]["ctrag_prefers_provider"] is True
    assert rankings["ctrag_recovery"][0] == "provider_recovered"
    assert metrics["ctrag_precision_at_1"] == 1.0
    assert metrics["terrain_only_precision_at_1"] == 0.0
    assert metrics["recency_precision_at_1"] == 0.0


def test_false_rescue_is_reported_beyond_top1(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    metrics = report["metrics"]

    assert metrics["false_rescue_at_1"] == 0.0
    assert 0.0 <= metrics["false_rescue_at_2"] <= 1.0
    assert 0.0 <= metrics["false_rescue_at_3"] <= 1.0
    assert 0.0 <= metrics["false_rescue_at_4"] <= 1.0
    assert metrics["false_rescue_at_2"] >= metrics["false_rescue_at_1"]


def test_structural_baseline_is_reachability_ceiling_not_global_ranker(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)
    structural = report["rankings"]["structural_recovered_unranked"]

    assert "provider_recovered" in structural
    assert "manual_recovered" in structural
    assert "script_recovered" in structural
    assert "cache_recovered" in structural
    assert "global_recovered" not in structural


def test_eroded_path_rescue_writes_machine_readable_artifacts(tmp_path) -> None:
    run(ErodedPathConfig(), tmp_path)
    assert (tmp_path / "eroded-path-rescue.json").exists()
    assert (tmp_path / "eroded-path-rescue.csv").exists()


def test_reachability_gate_is_filter_not_hidden_penalty() -> None:
    cfg = ErodedPathConfig()
    topology, edges = _build()

    # Mutation: change only feasibility. The globally strongest 9/10 recovery is
    # now reachable from the current retry state.
    global_reachable = _edge("retry", "global_fix")
    topology.add_edge(global_reachable)

    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(reinforcement_step=0.5, decay_rate=0.15, maximum_influence=30.0),
    )
    _observe_branch(
        terrain, edges["provider_1"], edges["provider_success"], edges["provider_failure"],
        cfg.provider_successes, cfg.provider_failures,
    )
    _observe_branch(
        terrain, edges["global_1"], edges["global_success"], edges["global_failure"],
        cfg.global_successes, cfg.global_failures,
    )
    # Observe the newly feasible entry without changing the terminal statistics.
    for _ in range(cfg.global_successes + cfg.global_failures):
        terrain.reinforce(global_reachable)

    result = TerrainAwareRetriever(CTRetriever(topology), terrain).search_staged(
        "payment recovery path",
        mode=QueryMode.RECOVERY,
        anchor_ids=["retry"],
        k=20,
    )
    recovered = [
        hit for hit in result.hits
        if hit.node.metadata.get("status") == "recovered"
    ]
    ranking = [hit.node.id for hit in recovered]

    assert ranking[0] == "global_recovered"
    global_hit = next(hit for hit in recovered if hit.node.id == "global_recovered")
    provider_hit = next(hit for hit in recovered if hit.node.id == "provider_recovered")
    assert global_hit.components["recovery_causally_reachable"] == 1.0
    assert provider_hit.components["recovery_causally_reachable"] == 1.0
    assert (
        global_hit.components["historical_wilson_lower_95"]
        > provider_hit.components["historical_wilson_lower_95"]
    )
