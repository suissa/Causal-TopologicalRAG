from __future__ import annotations

from ctrag.benchmarks.temporal_terrain_scenarios import (
    TemporalTerrainConfig,
    basin_attraction_drift,
    obsolete_healing_path_decay,
    out_of_order_causation_gap,
    run,
)


def test_obsolete_healing_path_erodes_without_deleting_history() -> None:
    result = obsolete_healing_path_decay(TemporalTerrainConfig())
    assert result["oracle_passed"] is True
    assert result["legacy_final_influence"] < result["legacy_peak_influence"]
    assert result["async_final_influence"] > result["legacy_final_influence"]
    assert result["async_recovered_rank"] < result["legacy_recovered_rank"]
    assert result["legacy_history_count"] == 10
    assert result["legacy_edge_still_present"] is True


def test_recurrent_failure_creates_new_scc_attractor_and_basin_drift() -> None:
    result = basin_attraction_drift(TemporalTerrainConfig())
    assert result["oracle_passed"] is True
    assert result["recurrent_scc_attractor"] is not None
    assert "A_compensation" in result["recurrent_scc_members"]
    assert result["basin_drift_mean"] > 0.0
    assert result["compensation_basin_probe_purity_after"] >= 0.75
    assert result["historical_success_edges_preserved"] is True


def test_out_of_order_causation_reconciles_across_large_clock_gap() -> None:
    result = out_of_order_causation_gap(TemporalTerrainConfig())
    assert result["oracle_passed"] is True
    assert result["pending_before_parent"] is True
    assert result["causal_edges_after_parent"] == 1
    assert result["causal_provenance"] == "event"
    assert result["causal_evidence_source"] == "event.causation_id"
    assert result["clock_gap_hours"] == 24 * 14
    assert result["parent_rank"] is not None
    assert result["parent_causal_component"] > 0.0


def test_temporal_terrain_scenario_runner_writes_artifacts(tmp_path) -> None:
    report = run(TemporalTerrainConfig(), tmp_path)
    assert report["all_oracles_passed"] is True
    assert (tmp_path / "temporal-terrain-scenarios.json").exists()
    assert (tmp_path / "temporal-terrain-scenarios.csv").exists()
