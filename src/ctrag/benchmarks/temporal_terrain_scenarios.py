"""Temporal deformation benchmark for CT-RAG terrain.

Three controlled scenarios:
1. obsolete healing path decay;
2. basin attraction drift under recurrent failure;
3. out-of-order causal reconciliation across a large event-time gap.

The benchmark preserves authoritative history. Terrain changes only navigational
influence; structural topology changes are explicit newly observed edges.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    DynamicTerrain,
    Edge,
    EdgeKind,
    EventProjector,
    EventRecord,
    MemoryNode,
    QueryMode,
    TerrainAwareRetriever,
    TerrainConfig,
)


@dataclass(frozen=True)
class TemporalTerrainConfig:
    legacy_repetitions: int = 10
    async_repetitions: int = 12
    decay_batches: int = 8
    decay_elapsed_per_batch: float = 2.0
    recurrent_failure_repetitions: int = 40
    causal_gap_hours: int = 24 * 14

    def __post_init__(self) -> None:
        for name in (
            "legacy_repetitions",
            "async_repetitions",
            "decay_batches",
            "recurrent_failure_repetitions",
            "causal_gap_hours",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.decay_elapsed_per_batch <= 0:
            raise ValueError("decay_elapsed_per_batch must be positive")


def _causal(source: str, target: str) -> Edge:
    return Edge(
        source=source,
        target=target,
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    )


def _rank_recovery(topology: CausalTopology, terrain: DynamicTerrain, anchor: str) -> list[str]:
    retriever = TerrainAwareRetriever(CTRetriever(topology), terrain)
    return [
        hit.node.id
        for hit in retriever.search(
            "network recovery route",
            mode=QueryMode.RECOVERY,
            anchor_ids=[anchor],
            exhaustive=True,
            k=6,
        )
    ]


def obsolete_healing_path_decay(config: TemporalTerrainConfig) -> dict[str, object]:
    topology = CausalTopology()
    for node_id, status in (
        ("NetworkTimeout", "failure"),
        ("LegacyRetryPath", "recovery"),
        ("LegacyRecovered", "recovered"),
        ("AsyncFallbackPath", "recovery"),
        ("AsyncRecovered", "recovered"),
    ):
        topology.add_node(
            MemoryNode(
                id=node_id,
                text="network recovery route",
                metadata={"status": status},
            )
        )

    legacy_1 = _causal("NetworkTimeout", "LegacyRetryPath")
    legacy_2 = _causal("LegacyRetryPath", "LegacyRecovered")
    async_1 = _causal("NetworkTimeout", "AsyncFallbackPath")
    async_2 = _causal("AsyncFallbackPath", "AsyncRecovered")
    for edge in (legacy_1, legacy_2, async_1, async_2):
        topology.add_edge(edge)

    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(
            reinforcement_step=0.45,
            decay_rate=0.18,
            minimum_influence=0.0,
            maximum_influence=20.0,
        ),
    )

    for _ in range(config.legacy_repetitions):
        terrain.reinforce(legacy_1)
        terrain.reinforce(legacy_2)

    legacy_peak = terrain.influence(legacy_1)

    decay_trace: list[dict[str, float | int]] = []
    for batch in range(config.decay_batches):
        terrain.decay(config.decay_elapsed_per_batch)
        terrain.reinforce(async_1)
        terrain.reinforce(async_2)
        if batch < config.async_repetitions - 1:
            # The number of new observations is bounded by the configured batches
            # here; additional observations are applied below if requested.
            pass
        decay_trace.append(
            {
                "batch": batch + 1,
                "legacy_influence": terrain.influence(legacy_1),
                "async_influence": terrain.influence(async_1),
            }
        )

    for _ in range(max(0, config.async_repetitions - config.decay_batches)):
        terrain.reinforce(async_1)
        terrain.reinforce(async_2)

    ranking = _rank_recovery(topology, terrain, "NetworkTimeout")
    legacy_rank = ranking.index("LegacyRecovered") + 1
    async_rank = ranking.index("AsyncRecovered") + 1

    return {
        "scenario": "obsolete_healing_path_decay",
        "legacy_peak_influence": legacy_peak,
        "legacy_final_influence": terrain.influence(legacy_1),
        "async_final_influence": terrain.influence(async_1),
        "legacy_history_count": terrain.transition_count(legacy_1),
        "async_history_count": terrain.transition_count(async_1),
        "legacy_edge_still_present": topology.has_edge(legacy_1),
        "async_edge_still_present": topology.has_edge(async_1),
        "ranking": ranking,
        "legacy_recovered_rank": legacy_rank,
        "async_recovered_rank": async_rank,
        "decay_trace": decay_trace,
        "oracle_passed": (
            terrain.influence(legacy_1) < legacy_peak
            and terrain.influence(async_1) > terrain.influence(legacy_1)
            and async_rank < legacy_rank
            and topology.has_edge(legacy_1)
            and terrain.transition_count(legacy_1) == config.legacy_repetitions
        ),
    }


def _probe_purity(members: set[str], probe: set[str]) -> float:
    return 0.0 if not probe else len(members & probe) / len(probe)


def basin_attraction_drift(config: TemporalTerrainConfig) -> dict[str, object]:
    topology = CausalTopology()
    for node_id in (
        "entry",
        "healthy_stage",
        "A_success",
        "degraded_stage",
        "A_compensation",
        "comp_loop",
    ):
        topology.add_node(MemoryNode(id=node_id, text=node_id))

    healthy_1 = _causal("entry", "healthy_stage")
    healthy_2 = _causal("healthy_stage", "A_success")
    topology.add_edge(healthy_1)
    topology.add_edge(healthy_2)
    topology.register_attractor("A_success", origin="benchmark:known-success")

    terrain = DynamicTerrain(
        topology,
        config=TerrainConfig(reinforcement_step=0.25, decay_rate=0.02),
    )
    for _ in range(20):
        terrain.reinforce(healthy_1)
        terrain.reinforce(healthy_2)

    before = terrain.snapshot()
    affected_probe = {"entry", "degraded_stage", "A_compensation", "comp_loop"}
    success_purity_before = _probe_purity(set(before.basins["A_success"]), affected_probe)

    # Newly observed degraded behavior changes the known topology at t1. Old
    # nodes/edges remain untouched.
    degraded_1 = _causal("entry", "degraded_stage")
    degraded_2 = _causal("degraded_stage", "A_compensation")
    cycle_1 = _causal("A_compensation", "comp_loop")
    cycle_2 = _causal("comp_loop", "A_compensation")
    for edge in (degraded_1, degraded_2, cycle_1, cycle_2):
        topology.add_edge(edge)

    # Structural SCC exists once the newly observed cycle is known, but before
    # repeated failure there is no empirical support for recurrent attraction.
    discovered_pre = terrain.discover_attractors(register=False)
    recurrent_pre = next(
        (
            item
            for item in discovered_pre
            if item.origin == "discovered:scc"
            and "A_compensation" in set(item.metadata.get("members", []))
        ),
        None,
    )
    recurrent_confidence_before = 0.0 if recurrent_pre is None else recurrent_pre.confidence
    recurrent_internal_before = (
        0 if recurrent_pre is None else int(recurrent_pre.metadata.get("internal_observations", 0))
    )

    for _ in range(config.recurrent_failure_repetitions):
        for edge in (degraded_1, degraded_2, cycle_1, cycle_2):
            terrain.reinforce(edge)

    discovered = terrain.discover_attractors(register=True)
    after = terrain.snapshot()
    drift = terrain.basin_drift(before, after)

    recurrent = next(
        (
            item
            for item in discovered
            if item.origin == "discovered:scc"
            and "A_compensation" in set(item.metadata.get("members", []))
        ),
        None,
    )
    recurrent_id = None if recurrent is None else recurrent.node_id
    compensation_basin = set() if recurrent_id is None else set(after.basins[recurrent_id])
    compensation_purity_after = _probe_purity(compensation_basin, affected_probe)
    recurrent_confidence_after = 0.0 if recurrent is None else recurrent.confidence
    recurrent_internal_after = (
        0 if recurrent is None else int(recurrent.metadata.get("internal_observations", 0))
    )
    problematic_membership_gain = (
        compensation_purity_after - success_purity_before
    )

    return {
        "scenario": "basin_attraction_drift",
        "before_attractors": sorted(before.attractors),
        "after_attractors": sorted(after.attractors),
        "recurrent_scc_attractor": recurrent_id,
        "recurrent_scc_members": [] if recurrent is None else recurrent.metadata.get("members", []),
        "basin_drift_mean": drift.mean,
        "basin_drift_per_attractor": drift.per_attractor,
        "affected_probe": sorted(affected_probe),
        "success_basin_probe_purity_before": success_purity_before,
        "compensation_basin_probe_purity_after": compensation_purity_after,
        "problematic_basin_membership_gain": problematic_membership_gain,
        "recurrent_confidence_before": recurrent_confidence_before,
        "recurrent_confidence_after": recurrent_confidence_after,
        "recurrent_internal_observations_before": recurrent_internal_before,
        "recurrent_internal_observations_after": recurrent_internal_after,
        "failure_cycle_observations": terrain.transition_count(cycle_1),
        "historical_success_edges_preserved": topology.has_edge(healthy_1) and topology.has_edge(healthy_2),
        "oracle_passed": (
            recurrent is not None
            and problematic_membership_gain > 0.0
            and recurrent_internal_after > recurrent_internal_before
            and recurrent_confidence_after > recurrent_confidence_before
            and terrain.transition_count(cycle_1) == config.recurrent_failure_repetitions
            and topology.has_edge(healthy_1)
        ),
    }


def out_of_order_causation_gap(config: TemporalTerrainConfig) -> dict[str, object]:
    topology = CausalTopology()
    projector = EventProjector(topology)
    base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    parent_event_time = base
    child_event_time = base + timedelta(hours=config.causal_gap_hours)

    child = EventRecord(
        event_id="Action_B_Success",
        event_type="Action.B.Success",
        timestamp=child_event_time,
        causation_id="Action_A_Intent",
        execution_id="exec-child",
        status="ok",
    )
    projector.ingest(child)
    causal_before_parent = topology.incoming("Action_B_Success", {EdgeKind.CAUSAL})

    # Arrival-time noise: unrelated events arrive before the delayed parent.
    for index in range(25):
        projector.ingest(
            EventRecord(
                event_id=f"noise-{index}",
                event_type="Action.B.Success.Context",
                timestamp=child_event_time + timedelta(minutes=index + 1),
                status="ok",
            )
        )

    parent = EventRecord(
        event_id="Action_A_Intent",
        event_type="Action.A.Intent",
        timestamp=parent_event_time,
        status="ok",
    )
    projector.ingest(parent)

    causal_after_parent = topology.incoming("Action_B_Success", {EdgeKind.CAUSAL})
    causal_edge = causal_after_parent[0] if causal_after_parent else None
    clock_gap_seconds = abs(
        (
            topology.nodes["Action_B_Success"].timestamp
            - topology.nodes["Action_A_Intent"].timestamp
        ).total_seconds()
    )

    retriever = CTRetriever(topology)
    query = "why did action B succeed"

    # Semantic-only baseline: global dense ranking, excluding the query anchor.
    semantic_ranked = [
        node_id
        for node_id, _ in retriever.rank_dense(query, k=len(topology.nodes))
        if node_id != "Action_B_Success"
    ]
    semantic_parent_rank = (
        semantic_ranked.index("Action_A_Intent") + 1
        if "Action_A_Intent" in semantic_ranked
        else None
    )

    # Recency-only baseline: newest event-time first, excluding the query anchor.
    recency_ranked = sorted(
        (node_id for node_id in topology.nodes if node_id != "Action_B_Success"),
        key=lambda node_id: (-topology.nodes[node_id].timestamp.timestamp(), node_id),
    )
    recency_parent_rank = recency_ranked.index("Action_A_Intent") + 1

    hits = retriever.search(
        query,
        mode=QueryMode.WHY,
        anchor_ids=["Action_B_Success"],
        exhaustive=True,
        k=10,
    )
    ranked = [hit.node.id for hit in hits]
    parent_rank = ranked.index("Action_A_Intent") + 1 if "Action_A_Intent" in ranked else None
    parent_hit = next((hit for hit in hits if hit.node.id == "Action_A_Intent"), None)

    top_k = 10
    semantic_parent_recalled_at_k = (
        semantic_parent_rank is not None and semantic_parent_rank <= top_k
    )
    recency_parent_recalled_at_k = recency_parent_rank <= top_k
    ctrag_parent_recalled_at_k = parent_rank is not None and parent_rank <= top_k

    return {
        "scenario": "out_of_order_causation_gap",
        "pending_before_parent": len(causal_before_parent) == 0,
        "causal_edges_after_parent": len(causal_after_parent),
        "causal_provenance": None if causal_edge is None else causal_edge.provenance.value,
        "causal_evidence_source": (
            None
            if causal_edge is None or not causal_edge.evidence
            else causal_edge.evidence[0].source
        ),
        "clock_gap_seconds": clock_gap_seconds,
        "clock_gap_hours": clock_gap_seconds / 3600.0,
        "noise_events_ingested": 25,
        "semantic_only_ranking_top10": semantic_ranked[:top_k],
        "recency_only_ranking_top10": recency_ranked[:top_k],
        "why_ranking": ranked,
        "semantic_parent_rank": semantic_parent_rank,
        "recency_parent_rank": recency_parent_rank,
        "parent_rank": parent_rank,
        "semantic_parent_recalled_at_10": semantic_parent_recalled_at_k,
        "recency_parent_recalled_at_10": recency_parent_recalled_at_k,
        "ctrag_parent_recalled_at_10": ctrag_parent_recalled_at_k,
        "parent_causal_component": None if parent_hit is None else parent_hit.components["causal"],
        "oracle_passed": (
            len(causal_before_parent) == 0
            and len(causal_after_parent) == 1
            and causal_edge is not None
            and causal_edge.provenance is CausalProvenance.EVENT
            and causal_edge.evidence[0].source == "event.causation_id"
            and clock_gap_seconds == config.causal_gap_hours * 3600
            and not semantic_parent_recalled_at_k
            and not recency_parent_recalled_at_k
            and ctrag_parent_recalled_at_k
            and parent_rank is not None
            and parent_hit is not None
            and parent_hit.components["causal"] > 0.0
        ),
    }


def run(config: TemporalTerrainConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [
        obsolete_healing_path_decay(config),
        basin_attraction_drift(config),
        out_of_order_causation_gap(config),
    ]
    report = {
        "schema_version": 1,
        "claim_scope": "controlled temporal-deformation mechanism tests",
        "config": asdict(config),
        "all_oracles_passed": all(bool(item["oracle_passed"]) for item in scenarios),
        "scenarios": scenarios,
    }
    (output / "temporal-terrain-scenarios.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for scenario in scenarios:
        rows.append(
            {
                "scenario": scenario["scenario"],
                "oracle_passed": scenario["oracle_passed"],
                "summary": json.dumps(
                    {key: value for key, value in scenario.items() if key not in {"decay_trace", "why_ranking", "semantic_only_ranking_top10", "recency_only_ranking_top10"}},
                    sort_keys=True,
                ),
            }
        )
    with (output / "temporal-terrain-scenarios.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run temporal terrain deformation scenarios")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/temporal-terrain-scenarios"),
    )
    args = parser.parse_args(argv)
    report = run(TemporalTerrainConfig(), args.output)
    print(json.dumps({
        "all_oracles_passed": report["all_oracles_passed"],
        "scenarios": {
            item["scenario"]: item["oracle_passed"]
            for item in report["scenarios"]
        },
    }, sort_keys=True))
    return 0 if report["all_oracles_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
