"""Chronological, deterministic terrain-drift experiment for issue #26.

This is a controlled mechanism study, not evidence of broad real-world benefit.
Each evaluation occurs before the next transition is observed, so future events
cannot affect an earlier score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ctrag import (
    CausalProvenance, CausalTopology, CTRetriever, DynamicTerrain, Edge,
    EdgeKind, MemoryNode, QueryMode, TerrainAwareRetriever, TerrainConfig,
)


@dataclass(frozen=True)
class TerrainDynamicsConfig:
    initial_repetitions: int = 8
    drift_repetitions: int = 8
    decay_elapsed: float = 1.0
    reset_drift_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.initial_repetitions < 1 or self.drift_repetitions < 1:
            raise ValueError("repetition counts must be positive")
        if self.decay_elapsed < 0:
            raise ValueError("decay_elapsed must be non-negative")
        if not 0 <= self.reset_drift_threshold <= 1:
            raise ValueError("reset_drift_threshold must be within [0, 1]")


def _mechanism() -> tuple[CausalTopology, Edge, Edge, Edge]:
    topology = CausalTopology()
    for node_id in ("root", "routine", "shifted", "critical"):
        topology.add_node(MemoryNode(id=node_id, text="next candidate"))
    routine = Edge("root", "routine", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)
    shifted = Edge("root", "shifted", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)
    critical = Edge("root", "critical", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)
    for edge in (routine, shifted, critical):
        topology.add_edge(edge)
    return topology, routine, shifted, critical


def _rank(terrain: DynamicTerrain) -> list[str]:
    return [
        hit.node.id
        for hit in TerrainAwareRetriever(CTRetriever(terrain.topology), terrain).search(
            "next candidate", mode=QueryMode.WHAT_NEXT, anchor_ids=["root"], exhaustive=True, k=3
        )
    ]


def run(config: TerrainDynamicsConfig, output: Path) -> dict:
    """Run a trajectory sequentially; every row is evaluated before its update."""
    output.mkdir(parents=True, exist_ok=True)
    topology, routine, shifted, critical = _mechanism()
    terrain = DynamicTerrain(topology, config=TerrainConfig(reinforcement_step=0.5, decay_rate=0.2))
    terrain.protect_edge(critical)
    rows: list[dict[str, object]] = []

    phases = (("initial", routine, config.initial_repetitions), ("drift", shifted, config.drift_repetitions))
    for phase, observed, repetitions in phases:
        for step in range(repetitions):
            ranked_before = _rank(terrain)
            rows.append({
                "phase": phase,
                "step": step,
                "observed_edge": f"{observed.source}->{observed.target}",
                "ranked_before_observation": ranked_before,
                "top1_before_observation": ranked_before[0],
                "critical_influence_before": terrain.influence(critical),
                "future_free": True,
            })
            terrain.observe_transition(observed.source, observed.target, provenance=observed.provenance)
            terrain.decay(config.decay_elapsed)

    before_reset = terrain.snapshot()
    # Basin membership did not change in this fixed mechanism: reset is driven by
    # measured ranking-regime drift, not an invented structural change.
    initial_top = next(row["top1_before_observation"] for row in rows if row["phase"] == "initial")
    final_top = rows[-1]["top1_before_observation"]
    ranking_regime_drift = 0.0 if initial_top == final_top else 1.0
    reset_recommended = ranking_regime_drift >= config.reset_drift_threshold
    if reset_recommended:
        terrain.reset_navigation()
    after_reset = terrain.snapshot()

    report = {
        "schema_version": 1,
        "claim_scope": "controlled chronological terrain mechanism only; not a broad superiority claim",
        "config": asdict(config),
        "rows": rows,
        "outcomes": {
            "initial_phase_final_top1": [row for row in rows if row["phase"] == "initial"][-1]["top1_before_observation"],
            "drift_phase_final_top1": final_top,
            "critical_floor_respected": min(float(row["critical_influence_before"]) for row in rows) >= terrain.config.protected_minimum_influence,
            "ranking_regime_drift": ranking_regime_drift,
            "reset_recommended": reset_recommended,
            "transition_history_preserved_after_reset": before_reset.transition_counts == after_reset.transition_counts,
            "overlay_cleared_after_reset": not after_reset.influences,
        },
        "protocol": {
            "chronological": "Each ranking is captured before observing the row transition.",
            "rare_critical_protection": "The critical edge is decay-floored; it is never promoted by unobserved labels.",
            "reset": "Reset clears only navigational influence, never topology or transition history.",
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "terrain-dynamics-results.json").write_text(encoded, encoding="utf-8", newline="\n")
    with (output / "terrain-dynamics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "ranked_before_observation": json.dumps(row["ranked_before_observation"])})
    manifest = {
        "schema_version": 1,
        "command": "python -m ctrag.benchmarks.terrain_dynamics",
        "results_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run chronological dynamic-terrain experiment")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/terrain-dynamics"))
    args = parser.parse_args(argv)
    report = run(TerrainDynamicsConfig(), args.output)
    print(f"Wrote {len(report['rows'])} chronological terrain observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
