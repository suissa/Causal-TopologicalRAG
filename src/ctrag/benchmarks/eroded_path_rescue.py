"""Controlled eroded-path rescue experiment.

Tests whether RECOVERY retrieval can surface a historically observed successful
path after DynamicTerrain has eroded its current navigational influence.

This is a mechanism benchmark, not a production superiority claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    DynamicTerrain,
    Edge,
    EdgeKind,
    MemoryNode,
    QueryMode,
    TerrainConfig,
)


@dataclass(frozen=True)
class ErodedPathConfig:
    failure_repetitions: int = 12
    recovery_prehistory_repetitions: int = 2
    decay_elapsed: float = 20.0
    terrain_weight: float = 1.0
    recovery_evidence_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.failure_repetitions < 1:
            raise ValueError("failure_repetitions must be positive")
        if self.recovery_prehistory_repetitions < 1:
            raise ValueError("recovery_prehistory_repetitions must be positive")
        if self.decay_elapsed < 0:
            raise ValueError("decay_elapsed must be non-negative")


def _build() -> tuple[CausalTopology, dict[str, Edge]]:
    topology = CausalTopology()
    nodes = {
        "retry": MemoryNode(id="retry", text="payment retry", metadata={"status": "retry"}),
        "timeout": MemoryNode(id="timeout", text="payment path", metadata={"status": "timeout"}),
        "human": MemoryNode(id="human", text="payment outcome", metadata={"status": "human_intervention"}),
        "fallback": MemoryNode(id="fallback", text="payment path", metadata={"status": "fallback"}),
        "recovered": MemoryNode(id="recovered", text="payment outcome", metadata={"status": "recovered"}),
    }
    for node in nodes.values():
        topology.add_node(node)

    edges = {
        "failure_1": Edge("retry", "timeout", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "failure_2": Edge("timeout", "human", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "recovery_1": Edge("retry", "fallback", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "recovery_2": Edge("fallback", "recovered", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
    }
    for edge in edges.values():
        topology.add_edge(edge)
    return topology, edges


def _terrain_path_score(terrain: DynamicTerrain, edges: tuple[Edge, ...]) -> float:
    score = 1.0
    for edge in edges:
        score *= terrain.influence(edge)
    return score ** (1.0 / len(edges))


def run(config: ErodedPathConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    topology, edges = _build()
    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(
            reinforcement_step=0.5,
            decay_rate=0.15,
            minimum_influence=0.0,
            maximum_influence=20.0,
        ),
    )

    recovery_path = (edges["recovery_1"], edges["recovery_2"])
    failure_path = (edges["failure_1"], edges["failure_2"])

    # The recovery path really happened in history.
    for _ in range(config.recovery_prehistory_repetitions):
        for edge in recovery_path:
            terrain.reinforce(edge)

    # It then falls out of use and erodes.
    terrain.decay(config.decay_elapsed)

    # Current operation repeatedly follows the degraded/failure basin.
    for _ in range(config.failure_repetitions):
        for edge in failure_path:
            terrain.reinforce(edge)

    failure_terrain = _terrain_path_score(terrain, failure_path)
    recovery_terrain = _terrain_path_score(terrain, recovery_path)

    # Terrain-only baseline: current navigational influence determines the route.
    terrain_only_winner = "failure" if failure_terrain >= recovery_terrain else "recovery"

    retriever = CTRetriever(topology)
    recovery_result = retriever.search_staged(
        "payment recovery path",
        mode=QueryMode.RECOVERY,
        anchor_ids=["retry"],
        k=5,
    )
    recovery_hits = recovery_result.hits
    rank = {hit.node.id: index + 1 for index, hit in enumerate(recovery_hits)}
    recovered_rank = rank.get("recovered")
    human_rank = rank.get("human")

    # Evidence-aware rescue policy: RECOVERY mode is allowed to consult preserved
    # causal history even when terrain influence is low. Terrain is reported as
    # context, not multiplied as a veto over historical recovery evidence.
    rescued = recovered_rank is not None and (human_rank is None or recovered_rank < human_rank)

    rows = [
        {
            "path": "failure",
            "terminal": "human",
            "terrain_influence": failure_terrain,
            "historical_observations": terrain.transition_count(edges["failure_1"]),
            "recovery_rank": human_rank,
        },
        {
            "path": "recovery",
            "terminal": "recovered",
            "terrain_influence": recovery_terrain,
            "historical_observations": terrain.transition_count(edges["recovery_1"]),
            "recovery_rank": recovered_rank,
        },
    ]

    report = {
        "schema_version": 1,
        "claim_scope": "controlled eroded-path rescue mechanism only",
        "config": {
            "failure_repetitions": config.failure_repetitions,
            "recovery_prehistory_repetitions": config.recovery_prehistory_repetitions,
            "decay_elapsed": config.decay_elapsed,
        },
        "rows": rows,
        "outcomes": {
            "terrain_only_winner": terrain_only_winner,
            "recovery_path_is_eroded": recovery_terrain < failure_terrain,
            "recovery_path_preserved_in_topology": topology.has_edge(edges["recovery_1"]) and topology.has_edge(edges["recovery_2"]),
            "recovery_path_has_observed_history": terrain.transition_count(edges["recovery_1"]) > 0,
            "ctrag_recovery_rank_recovered": recovered_rank,
            "ctrag_recovery_rank_failure_terminal": human_rank,
            "ctrag_rescues_eroded_recovery_path": rescued,
        },
        "interpretation": (
            "Terrain-only navigation follows current influence. RECOVERY retrieval may "
            "surface an observed historical recovery path despite low current influence; "
            "this is retrieval of preserved evidence, not a counterfactual causal proof."
        ),
    }

    (output / "eroded-path-rescue.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "eroded-path-rescue.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CT-RAG eroded-path rescue experiment")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/eroded-path-rescue"))
    args = parser.parse_args(argv)
    report = run(ErodedPathConfig(), args.output)
    print(json.dumps(report["outcomes"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
