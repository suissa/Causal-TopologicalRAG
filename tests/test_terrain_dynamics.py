from __future__ import annotations

import json

from ctrag.benchmarks.terrain_dynamics import TerrainDynamicsConfig, run


def test_dynamic_terrain_experiment_is_chronological_and_reproducible(tmp_path) -> None:
    first = run(TerrainDynamicsConfig(initial_repetitions=3, drift_repetitions=3), tmp_path / "first")
    second = run(TerrainDynamicsConfig(initial_repetitions=3, drift_repetitions=3), tmp_path / "second")

    assert first == second
    assert all(row["future_free"] for row in first["rows"])
    assert first["outcomes"]["critical_floor_respected"]
    assert first["outcomes"]["transition_history_preserved_after_reset"]
    assert first["outcomes"]["overlay_cleared_after_reset"]
    assert first["outcomes"]["reset_recommended"]
    assert (tmp_path / "first" / "manifest.json").exists()
    assert json.loads((tmp_path / "first" / "terrain-dynamics-results.json").read_text()) == first
