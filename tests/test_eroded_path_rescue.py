from __future__ import annotations

from ctrag.benchmarks.eroded_path_rescue import ErodedPathConfig, run


def test_eroded_path_rescue_separates_current_terrain_from_preserved_recovery(tmp_path) -> None:
    report = run(ErodedPathConfig(), tmp_path)

    outcomes = report["outcomes"]
    assert outcomes["recovery_path_is_eroded"] is True
    assert outcomes["terrain_only_winner"] == "failure"
    assert outcomes["recovery_path_preserved_in_topology"] is True
    assert outcomes["recovery_path_has_observed_history"] is True
    assert outcomes["ctrag_rescues_eroded_recovery_path"] is True
    assert outcomes["ctrag_recovery_rank_recovered"] < outcomes["ctrag_recovery_rank_failure_terminal"]


def test_eroded_path_rescue_writes_machine_readable_artifacts(tmp_path) -> None:
    run(ErodedPathConfig(), tmp_path)

    assert (tmp_path / "eroded-path-rescue.json").exists()
    assert (tmp_path / "eroded-path-rescue.csv").exists()
