from __future__ import annotations

import json
from pathlib import Path

from ctrag.benchmarks.runner import Config, run
from ctrag.research_artifacts import generate


def test_research_artifacts_are_generated_from_machine_outputs(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark-results"
    artifact_dir = tmp_path / "research-artifacts"
    run(Config(seeds=(7,), ks=(3,), traces=1, dimensions=64, max_hops=8), benchmark_dir)

    manifest = generate(benchmark_dir, artifact_dir, k=3)

    expected = {
        "topology.dot",
        "causal-path.dot",
        "benchmark-k3.md",
        "README.md",
        "manifest.json",
    }
    assert {path.name for path in artifact_dir.iterdir()} == expected
    assert "digraph CTRAGTopology" in (artifact_dir / "topology.dot").read_text(encoding="utf-8")
    assert "digraph CausalPath" in (artifact_dir / "causal-path.dot").read_text(encoding="utf-8")

    table = (artifact_dir / "benchmark-k3.md").read_text(encoding="utf-8")
    assert "recall_at_k" in table
    assert "full_ctrag" in table
    assert "dense_only" in table

    persisted = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest
    assert persisted["benchmark_command"] == "python -m ctrag.benchmarks"
    assert set(persisted["inputs_sha256"]) == {"config", "datasets", "table"}
    assert set(persisted["outputs_sha256"]) == {
        "topology", "causal_path", "benchmark_table", "readme"
    }


def test_research_artifact_generation_is_deterministic(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark-results"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    run(Config(seeds=(42,), ks=(3,), traces=1, dimensions=32), benchmark_dir)

    first = generate(benchmark_dir, first_dir, k=3)
    second = generate(benchmark_dir, second_dir, k=3)

    assert first["inputs_sha256"] == second["inputs_sha256"]
    assert first["outputs_sha256"] == second["outputs_sha256"]
    for filename in ("topology.dot", "causal-path.dot", "benchmark-k3.md", "README.md"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
