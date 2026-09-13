"""Gate D falsification experiments.

This module is deliberately restricted to generated train/dev mechanisms.  It
never imports or materializes the sealed final-test split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ctrag.embedding import HashingEmbedder
from ctrag.models import CausalProvenance, Edge, EdgeKind, QueryMode
from ctrag.retriever import CTRetriever
from ctrag.topology import CausalTopology

from .datasets import Dataset, Query, generate
from .metrics import evaluate
from .protocol import preregistration_manifest

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MODES = (QueryMode.WHY, QueryMode.WHAT_NEXT, QueryMode.RECOVERY)


def _edges(topology: CausalTopology) -> list[Edge]:
    return [edge for node_id in sorted(topology.nodes) for edge in topology.outgoing(node_id)]


def clone_topology(dataset: Dataset, edges: Iterable[Edge] | None = None) -> CausalTopology:
    result = CausalTopology()
    for node in dataset.topology.nodes.values():
        result.add_node(copy.deepcopy(node))
    seen = set()
    for edge in _edges(dataset.topology) if edges is None else edges:
        if edge.identity() in seen:
            continue
        seen.add(edge.identity())
        result.add_edge(copy.deepcopy(edge))
    return result


def _edge(edge: Edge, source: str, target: str, **changes: object) -> Edge:
    return replace(edge, source=source, target=target, **changes)


def destroy_topology(dataset: Dataset, control: str, seed: int) -> CausalTopology:
    """Return a content-identical topology negative control."""
    rng = random.Random(seed)
    edges = _edges(dataset.topology)
    if control == "intact":
        return clone_topology(dataset, edges)
    if control == "remove_causal":
        return clone_topology(dataset, [e for e in edges if e.kind is not EdgeKind.CAUSAL])
    if control == "remove_temporal":
        return clone_topology(dataset, [e for e in edges if e.kind is not EdgeKind.TEMPORAL])
    if control == "random_direction":
        return clone_topology(dataset, [_edge(e, e.target, e.source) if rng.random() < .5 else e for e in edges])
    if control == "degree_preserving_permutation":
        # Permuting node labels is an exact directed-degree-preserving edge swap.
        node_ids = sorted(dataset.topology.nodes)
        shuffled = list(node_ids)
        rng.shuffle(shuffled)
        mapping = dict(zip(node_ids, shuffled))
        return clone_topology(dataset, [_edge(e, mapping[e.source], mapping[e.target]) for e in edges])
    if control == "causation_permutation":
        causal = [e for e in edges if e.kind is EdgeKind.CAUSAL]
        targets = [e.target for e in causal]
        rng.shuffle(targets)
        changed = [_edge(e, e.source, target) for e, target in zip(causal, targets)]
        return clone_topology(dataset, [e for e in edges if e.kind is not EdgeKind.CAUSAL] + changed)
    if control == "topology_only_sham":
        # Same edge count and edge kinds, but endpoints are unrelated to execution.
        nodes = sorted(dataset.topology.nodes)
        changed: list[Edge] = []
        occupied: set[tuple[str, str, EdgeKind, CausalProvenance | None]] = set()
        for edge in edges:
            for _ in range(len(nodes) ** 2):
                source, target = rng.sample(nodes, 2)
                candidate = _edge(edge, source, target)
                if candidate.identity() not in occupied:
                    occupied.add(candidate.identity())
                    changed.append(candidate)
                    break
        return clone_topology(dataset, changed)
    raise ValueError(f"unknown destruction control: {control}")


def corrupt_causal(dataset: Dataset, kind: str, level: float, seed: int) -> CausalTopology:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    rng = random.Random(seed)
    edges = _edges(dataset.topology)
    causal = [e for e in edges if e.kind is EdgeKind.CAUSAL]
    other = [e for e in edges if e.kind is not EdgeKind.CAUSAL]
    n = round(level * len(causal))
    selected = set(rng.sample(range(len(causal)), n))
    if kind in {"missing", "removed"}:
        causal = [e for i, e in enumerate(causal) if i not in selected]
    elif kind == "incorrect_direction":
        causal = [_edge(e, e.target, e.source) if i in selected else e for i, e in enumerate(causal)]
    elif kind == "low_confidence":
        causal = [replace(e, confidence=max(0.01, 1.0 - level)) for e in causal]
    elif kind == "noisy":
        nodes = sorted(dataset.topology.nodes)
        occupied = {e.identity() for e in edges}
        additions = []
        for _ in range(n):
            for _ in range(len(nodes) ** 2):
                source, target = rng.sample(nodes, 2)
                candidate = Edge(source, target, EdgeKind.CAUSAL,
                                 provenance=CausalProvenance.HYPOTHESIZED, confidence=.35)
                if candidate.identity() not in occupied:
                    occupied.add(candidate.identity())
                    additions.append(candidate)
                    break
        causal += additions
    elif kind == "mixed_provenance":
        causal = [replace(e, provenance=(CausalProvenance.INFERRED if i in selected
                                        else CausalProvenance.EXECUTION))
                  for i, e in enumerate(causal)]
    else:
        raise ValueError(f"unknown corruption: {kind}")
    return clone_topology(dataset, other + causal)


def _run_queries(dataset: Dataset, topology: CausalTopology, *, weighted: bool = True) -> list[dict]:
    if not weighted:
        # Normalize metadata to the strongest class to provide the requested
        # unweighted comparator without changing endpoints.
        edges = [replace(e, provenance=CausalProvenance.EXECUTION, confidence=1.0)
                 if e.kind is EdgeKind.CAUSAL else e for e in _edges(topology)]
        topology = clone_topology(dataset, edges)
    retriever = CTRetriever(topology, embedder=HashingEmbedder(256))
    rows = []
    for query in dataset.queries:
        hits = retriever.search(query.text, mode=query.mode, anchor_ids=[query.anchor],
                                k=3, max_hops=8, exhaustive=True)
        metrics = evaluate([hit.node.id for hit in hits], query, dataset.topology, 3)
        relevant_hits = [hit for hit in hits if hit.node.id in query.causal_nodes]
        metrics["causal_evidence_rate"] = (
            sum(hit.causal_path is not None for hit in relevant_hits) / len(relevant_hits)
            if relevant_hits else 0.0
        )
        rows.append({"query_id": query.id, "mode": query.mode.value, **metrics})
    return rows


def _means(rows: list[dict]) -> dict[str, float | None]:
    keys = ("recall_at_k", "ndcg", "causal_path_recall", "causal_distance_error", "causal_evidence_rate")
    return {key: (statistics.mean(values) if (values := [r[key] for r in rows if r[key] is not None]) else None)
            for key in keys}


def topology_placebos(seeds: tuple[int, ...] = (7, 42, 2024)) -> list[dict]:
    controls = ("intact", "degree_preserving_permutation", "remove_causal", "remove_temporal",
                "random_direction", "causation_permutation", "topology_only_sham")
    raw: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for seed in seeds:
        for name in ("failure_recovery", "branching"):
            dataset = generate(name, seed)
            for control in controls:
                for row in _run_queries(dataset, destroy_topology(dataset, control, seed + 991)):
                    raw[(control, row["mode"])].append(row)
    intact = {mode: _means(rows) for (control, mode), rows in raw.items() if control == "intact"}
    result = []
    for (control, mode), rows in sorted(raw.items()):
        metrics = _means(rows)
        result.append({"control": control, "mode": mode, **metrics,
                       "recall_effect_vs_intact": metrics["recall_at_k"] - intact[mode]["recall_at_k"],
                       "causal_path_effect_vs_intact": metrics["causal_path_recall"] - intact[mode]["causal_path_recall"],
                       "causal_evidence_effect_vs_intact": metrics["causal_evidence_rate"] - intact[mode]["causal_evidence_rate"]})
    return result


def robustness_curves(seeds: tuple[int, ...] = (7, 42, 2024)) -> list[dict]:
    result = []
    for corruption in ("missing", "removed", "noisy", "incorrect_direction", "low_confidence", "mixed_provenance"):
        for level in LEVELS:
            buckets: dict[tuple[str, bool], list[dict]] = defaultdict(list)
            for seed in seeds:
                for name in ("failure_recovery", "branching"):
                    dataset = generate(name, seed)
                    topology = corrupt_causal(dataset, corruption, level, seed + 1776)
                    for weighted in (True, False):
                        for row in _run_queries(dataset, topology, weighted=weighted):
                            buckets[(row["mode"], weighted)].append(row)
            for (mode, weighted), rows in sorted(buckets.items()):
                result.append({"corruption": corruption, "level": level, "mode": mode,
                               "provenance_weighted": weighted, **_means(rows)})
    return result


class FutureLeakageError(RuntimeError):
    pass


def chronological_snapshot(dataset: Dataset, cutoff: datetime) -> CausalTopology:
    visible = {node.id for node in dataset.topology.nodes.values() if node.timestamp <= cutoff}
    result = CausalTopology()
    for node_id in sorted(visible):
        result.add_node(copy.deepcopy(dataset.topology.nodes[node_id]))
    for edge in _edges(dataset.topology):
        if edge.source in visible and edge.target in visible:
            result.add_edge(copy.deepcopy(edge))
    assert_no_future_visibility(result, cutoff)
    return result


def assert_no_future_visibility(topology: CausalTopology, cutoff: datetime) -> None:
    future = sorted(node.id for node in topology.nodes.values() if node.timestamp > cutoff)
    if future:
        raise FutureLeakageError(f"future nodes visible at cutoff: {future}")
    dangling = [(e.source, e.target) for e in _edges(topology)
                if e.source not in topology.nodes or e.target not in topology.nodes]
    if dangling:
        raise FutureLeakageError(f"future/dangling edges visible at cutoff: {dangling}")


def prospective_evaluation(seeds: tuple[int, ...] = (7, 42, 2024)) -> list[dict]:
    rows = []
    for seed in seeds:
        dataset = generate("failure_recovery", seed)
        for query in dataset.queries:
            if query.mode not in {QueryMode.WHAT_NEXT, QueryMode.RECOVERY}:
                continue
            cutoff = dataset.topology.nodes[query.anchor].timestamp
            snapshot = chronological_snapshot(dataset, cutoff)
            retriever = CTRetriever(snapshot, embedder=HashingEmbedder(256))
            hits = retriever.search(query.text, mode=query.mode, anchor_ids=[query.anchor], k=3,
                                    max_hops=8, exhaustive=True)
            future_gold_texts = {dataset.topology.nodes[node_id].text for node_id in query.relevance}
            analogue_matches = sum(hit.node.text in future_gold_texts for hit in hits)
            # Future gold is never inserted. Score only historical analogue utility;
            # an empty intersection is an honest negative result.
            visible_gold = {node_id: grade for node_id, grade in query.relevance.items()
                            if node_id in snapshot.nodes}
            visible_query = replace(query, relevance=visible_gold,
                                    causal_nodes=[n for n in query.causal_nodes if n in snapshot.nodes],
                                    causal_paths=[], causal_distances={})
            metrics = evaluate([hit.node.id for hit in hits], visible_query, snapshot, 3)
            rows.append({"dataset": dataset.name, "seed": seed, "query_id": query.id,
                         "mode": query.mode.value, "evaluation": "prospective",
                         "cutoff": cutoff.isoformat(), "visible_nodes": len(snapshot.nodes),
                         "future_nodes_excluded": len(dataset.topology.nodes) - len(snapshot.nodes),
                         "prospective_analogue_recall_at_3": analogue_matches / max(1, len(future_gold_texts)),
                         **metrics})
    return rows


def evidence_contract_artifact() -> dict:
    return {
        "schema_version": 1,
        "levels": {
            "observational_support": "association, temporal order, or historical divergence only",
            "interventional_evidence": "known randomized or simulated intervention with ground-truth target",
            "counterfactual_ground_truth": "paired potential outcomes or identified SCM counterfactual label",
        },
        "ordinary_event_logs_maximum_level": "observational_support",
        "headline_counterfactual_requirement": "counterfactual_ground_truth",
        "forbidden_in_observational_outputs": ["identified causal effect", "would have caused", "proved causality"],
        "confounded_negative_control": {
            "name": "alarm_and_failure_common_cause",
            "structure": "latent load -> alarm; latent load -> failure; alarm does not cause failure",
            "expected_level": "observational_support",
            "interventional_claim_permitted": False,
        },
        "interventional_evaluation": {
            "name": "simulated_binary_scm",
            "structural_equations": "Y := X xor U; do(X=0/1), U observed by simulator",
            "ground_truth_available": True,
            "reported_separately": True,
        },
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    before = preregistration_manifest()
    payload = {
        "schema_version": 1,
        "scope": "exploratory Gate D on generated train/dev mechanisms; final holdout not loaded",
        "seeds": [7, 42, 2024], "k": 3,
        "topology_placebos": topology_placebos(),
        "robustness_curves": robustness_curves(),
        "evidence_contract": evidence_contract_artifact(),
        "prospective": prospective_evaluation(),
        "preregistration": before,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output / "gate-d-results.json").write_text(encoded, encoding="utf-8", newline="\n")
    manifest = {"schema_version": 1, "artifact": "gate-d-results.json",
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "final_holdout_status": "sealed_not_loaded",
                "preregistration_sha256": before["sha256"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gate-d-results"))
    args = parser.parse_args()
    result = run(args.output)
    print(f"Gate D complete: {len(result['topology_placebos'])} placebo cells, "
          f"{len(result['robustness_curves'])} robustness cells, "
          f"{len(result['prospective'])} prospective queries")


if __name__ == "__main__":
    main()
