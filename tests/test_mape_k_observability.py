from __future__ import annotations

import json

from ctrag.benchmarks.mape_k_observability import MapeKObservabilityConfig, run


def test_mape_k_observability_experiment_is_reproducible_and_evidence_bounded(tmp_path) -> None:
    config = MapeKObservabilityConfig(healthy_cycles=1, recovered_cycles=1)
    first = run(config, tmp_path / "first")
    second = run(config, tmp_path / "second")

    assert first == second
    assert first["topology"]["authoritative_event_history_preserved"]
    assert first["topology"]["observed_event_causal_edge_count"] > 0
    assert all(first["outcomes"].values())
    assert (tmp_path / "first" / "events.ndjson").read_text().count("\n") == first["event_count"]
    assert (tmp_path / "first" / "metrics.csv").read_text().count("\n") == 4
    assert json.loads((tmp_path / "first" / "mape-k-observability-results.json").read_text()) == first
    assert json.loads((tmp_path / "first" / "manifest.json").read_text())["results_sha256"]
