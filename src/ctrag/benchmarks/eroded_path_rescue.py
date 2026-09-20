"""Controlled eroded-path rescue experiment with adversarial distractors.

The benchmark asks a non-trivial question: when multiple historical paths are
eroded, can RECOVERY rescue the path with strong successful evidence without also
rescuing an eroded low-success distractor?

Baselines:
- terrain-only: current navigational influence;
- recency: most recent historically successful terminal;
- structural ceiling: all historical paths that can reach a recovered terminal,
  without ranking by evidence quality.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    TerrainAwareRetriever,
    TerrainConfig,
)


@dataclass(frozen=True)
class ErodedPathConfig:
    failure_repetitions: int = 20
    provider_successes: int = 8
    manual_successes: int = 1
    manual_failures: int = 9
    decay_elapsed: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "failure_repetitions",
            "provider_successes",
            "manual_successes",
            "manual_failures",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.decay_elapsed < 0:
            raise ValueError("decay_elapsed must be non-negative")


def _build() -> tuple[CausalTopology, dict[str, Edge]]:
    topology = CausalTopology()
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    nodes = {
        "retry": MemoryNode(id="retry", text="payment retry", timestamp=now, metadata={"status": "retry"}),
        "timeout": MemoryNode(id="timeout", text="payment path", timestamp=now, metadata={"status": "timeout"}),
        "human": MemoryNode(id="human", text="payment outcome", timestamp=now, metadata={"status": "human_intervention"}),
        "provider": MemoryNode(id="provider", text="payment recovery route", timestamp=now - timedelta(days=30), metadata={"status": "fallback"}),
        "provider_recovered": MemoryNode(
            id="provider_recovered",
            text="payment recovered",
            timestamp=now - timedelta(days=30),
            metadata={"status": "recovered", "recovery_method": "provider_fallback"},
        ),
        "manual": MemoryNode(id="manual", text="payment recovery route", timestamp=now - timedelta(days=5), metadata={"status": "manual_patch"}),
        "manual_recovered": MemoryNode(
            id="manual_recovered",
            text="payment recovered",
            timestamp=now - timedelta(days=5),
            metadata={"status": "recovered", "recovery_method": "manual_patch"},
        ),
        "manual_failed": MemoryNode(
            id="manual_failed",
            text="payment failed",
            timestamp=now - timedelta(days=4),
            metadata={"status": "failed", "recovery_method": "manual_patch"},
        ),
    }
    for node in nodes.values():
        topology.add_node(node)

    edges = {
        "failure_1": Edge("retry", "timeout", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "failure_2": Edge("timeout", "human", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "provider_1": Edge("retry", "provider", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "provider_success": Edge("provider", "provider_recovered", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "manual_1": Edge("retry", "manual", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "manual_success": Edge("manual", "manual_recovered", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
        "manual_failure": Edge("manual", "manual_failed", EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION),
    }
    for edge in edges.values():
        topology.add_edge(edge)
    return topology, edges


def _terrain_path_score(terrain: DynamicTerrain, edges: tuple[Edge, ...]) -> float:
    score = 1.0
    for edge in edges:
        score *= terrain.influence(edge)
    return score ** (1.0 / len(edges))


def _structural_recovered_terminals(topology: CausalTopology, anchor: str) -> list[str]:
    result = []
    for node_id, node in topology.nodes.items():
        if node_id == anchor:
            continue
        if str(node.metadata.get("status", "")).lower() != "recovered":
            continue
        path = topology.causal_path_evidence(anchor, node_id, direction="out", max_hops=4)
        if path is not None:
            result.append(node_id)
    return sorted(result)


def _recency_recovered_ranking(topology: CausalTopology, terminals: list[str]) -> list[str]:
    return sorted(
        terminals,
        key=lambda node_id: (-topology.nodes[node_id].timestamp.timestamp(), node_id),
    )


def _precision_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    selected = ranking[:k]
    if not selected:
        return 0.0
    return sum(node_id in gold for node_id in selected) / len(selected)


def run(config: ErodedPathConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    topology, edges = _build()
    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(
            reinforcement_step=0.5,
            decay_rate=0.15,
            minimum_influence=0.0,
            maximum_influence=30.0,
        ),
    )

    provider_path = (edges["provider_1"], edges["provider_success"])
    manual_success_path = (edges["manual_1"], edges["manual_success"])
    failure_path = (edges["failure_1"], edges["failure_2"])

    # High-quality historical recovery: 8/8 successful ProviderFallback outcomes.
    for _ in range(config.provider_successes):
        terrain.reinforce(edges["provider_1"])
        terrain.reinforce(edges["provider_success"])

    # Low-quality historical recovery: ManualPatch succeeds rarely and fails often.
    for _ in range(config.manual_successes):
        terrain.reinforce(edges["manual_1"])
        terrain.reinforce(edges["manual_success"])
    for _ in range(config.manual_failures):
        terrain.reinforce(edges["manual_1"])
        terrain.reinforce(edges["manual_failure"])

    # Both historical alternatives become stale/eroded.
    terrain.decay(config.decay_elapsed)

    # Current dominant basin repeatedly ends in human intervention.
    for _ in range(config.failure_repetitions):
        terrain.reinforce(edges["failure_1"])
        terrain.reinforce(edges["failure_2"])

    provider_terrain = _terrain_path_score(terrain, provider_path)
    manual_terrain = _terrain_path_score(terrain, manual_success_path)
    failure_terrain = _terrain_path_score(terrain, failure_path)

    terrain_only_ranking = sorted(
        [
            ("human", failure_terrain),
            ("provider_recovered", provider_terrain),
            ("manual_recovered", manual_terrain),
        ],
        key=lambda item: (-item[1], item[0]),
    )
    terrain_only = [item[0] for item in terrain_only_ranking]

    structural = _structural_recovered_terminals(topology, "retry")
    recency = _recency_recovered_ranking(topology, structural)

    result = TerrainAwareRetriever(CTRetriever(topology), terrain).search_staged(
        "payment recovery path",
        mode=QueryMode.RECOVERY,
        anchor_ids=["retry"],
        k=8,
    )
    ctrag_ranking = [hit.node.id for hit in result.hits if hit.node.id in set(structural)]
    components = {
        hit.node.id: {
            "score": hit.score,
            "terrain_influence": hit.components.get("terrain_influence", 1.0),
            "historical_success_rate": hit.components.get("historical_success_rate", 0.0),
            "historical_support": hit.components.get("historical_support", 0.0),
        }
        for hit in result.hits
        if hit.node.id in set(structural)
    }

    gold = {"provider_recovered"}
    provider_success_rate = config.provider_successes / config.provider_successes
    manual_success_rate = config.manual_successes / (config.manual_successes + config.manual_failures)

    metrics = {
        "terrain_only_precision_at_1": _precision_at_k(terrain_only, gold, 1),
        "recency_precision_at_1": _precision_at_k(recency, gold, 1),
        "structural_precision_at_1": _precision_at_k(structural, gold, 1),
        "structural_precision_at_2": _precision_at_k(structural, gold, 2),
        "ctrag_precision_at_1": _precision_at_k(ctrag_ranking, gold, 1),
        "ctrag_false_rescue_at_1": 1.0 - _precision_at_k(ctrag_ranking, gold, 1),
    }

    rows = [
        {
            "path": "dominant_failure",
            "terminal": "human",
            "terrain_influence": failure_terrain,
            "historical_success_rate": 0.0,
            "observations": config.failure_repetitions,
            "last_success_recency_days": None,
        },
        {
            "path": "provider_fallback",
            "terminal": "provider_recovered",
            "terrain_influence": provider_terrain,
            "historical_success_rate": provider_success_rate,
            "observations": config.provider_successes,
            "last_success_recency_days": 30,
        },
        {
            "path": "manual_patch",
            "terminal": "manual_recovered",
            "terrain_influence": manual_terrain,
            "historical_success_rate": manual_success_rate,
            "observations": config.manual_successes + config.manual_failures,
            "last_success_recency_days": 5,
        },
    ]

    report = {
        "schema_version": 2,
        "claim_scope": "controlled eroded-path rescue discrimination mechanism",
        "config": {
            "failure_repetitions": config.failure_repetitions,
            "provider_successes": config.provider_successes,
            "manual_successes": config.manual_successes,
            "manual_failures": config.manual_failures,
            "decay_elapsed": config.decay_elapsed,
        },
        "rows": rows,
        "rankings": {
            "terrain_only": terrain_only,
            "recency_recovered_only": recency,
            "structural_recovered_unranked": structural,
            "ctrag_recovery": ctrag_ranking,
        },
        "ctrag_components": components,
        "metrics": metrics,
        "outcomes": {
            "provider_path_eroded_vs_failure": provider_terrain < failure_terrain,
            "manual_path_eroded_vs_failure": manual_terrain < failure_terrain,
            "provider_history_preserved": terrain.transition_count(edges["provider_success"]) == config.provider_successes,
            "manual_history_preserved": (
                terrain.transition_count(edges["manual_success"]) == config.manual_successes
                and terrain.transition_count(edges["manual_failure"]) == config.manual_failures
            ),
            "provider_success_rate": provider_success_rate,
            "manual_success_rate": manual_success_rate,
            "recency_prefers_low_success_distractor": bool(recency and recency[0] == "manual_recovered"),
            "structural_baseline_contains_both_recoveries": set(structural) == {"provider_recovered", "manual_recovered"},
            "ctrag_prefers_high_success_eroded_path": bool(ctrag_ranking and ctrag_ranking[0] == "provider_recovered"),
            "ctrag_does_not_false_rescue_manual_at_1": metrics["ctrag_false_rescue_at_1"] == 0.0,
        },
        "interpretation": (
            "Low terrain influence is not a veto in RECOVERY mode. Historical recovery "
            "candidates are ordered by observed branch success and support. The benchmark "
            "includes a more recent but low-success eroded distractor, so success requires "
            "discrimination rather than merely ignoring terrain."
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
    print(json.dumps({
        "rankings": report["rankings"],
        "metrics": report["metrics"],
        "outcomes": report["outcomes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
