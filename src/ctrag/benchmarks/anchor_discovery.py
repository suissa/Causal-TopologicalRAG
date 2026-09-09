from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ctrag.adapters import RankBM25Retriever, SentenceTransformersEmbedder, reciprocal_rank_fusion
from ctrag.directional_retriever import CTRetriever
from .datasets import generate
from .holdout import load_holdout_spec, split_dataset_sha256, split_seeds
from .metrics import evaluate
from .protocol import preregistration_manifest
from .strong_baselines import CachedEmbedder, STRONG_DENSE_MODELS

ANCHOR_MODEL = STRONG_DENSE_MODELS[0]
ANCHOR_K = 3


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rankings(retriever: CTRetriever, query_text: str) -> dict[str, list[str]]:
    size = len(retriever.topology.nodes)
    semantic = [node_id for node_id, _ in retriever.rank_dense(query_text, k=size)]
    lexical = [node_id for node_id, _ in retriever.rank_lexical(query_text, k=size)]
    hybrid = [node_id for node_id, _ in reciprocal_rank_fusion([semantic, lexical], rank_constant=60)]
    return {"semantic": semantic, "lexical": lexical, "hybrid": hybrid}


def _ambiguous_queries(topology) -> list[dict[str, Any]]:
    """Create multi-gold stress queries without changing the frozen dataset manifest."""
    groups: dict[str, list[str]] = defaultdict(list)
    for node in topology.nodes.values():
        lowered = node.text.lower()
        if lowered.endswith("operation failed"):
            groups["operation failed"].append(node.id)
        elif lowered == "request accepted":
            groups["request accepted"].append(node.id)
    return [
        {"text": text, "gold_anchor_ids": sorted(ids)}
        for text, ids in sorted(groups.items())
        if len(ids) >= 2
    ]


def _summarize_anchor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["selector"])].append(row)
    summary = []
    for (dataset, selector), group in sorted(grouped.items()):
        top1 = [float(row["anchor_top1_correct"]) for row in group]
        topk = [float(row["anchor_topk_correct"]) for row in group]
        summary.append({
            "dataset": dataset,
            "selector": selector,
            "n": len(group),
            "anchor_top1_accuracy": statistics.mean(top1),
            "anchor_top1_std": statistics.stdev(top1) if len(top1) > 1 else 0.0,
            "anchor_top3_accuracy": statistics.mean(topk),
            "anchor_top3_std": statistics.stdev(topk) if len(topk) > 1 else 0.0,
        })
    return summary


def _summarize_downstream(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    metric_names = (
        "recall_at_k",
        "mrr",
        "ndcg",
        "causal_path_recall",
        "causal_distance_error",
        "context_token_efficiency",
    )
    for row in rows:
        grouped[(row["dataset"], row["selector"], row["track"], row["k"])].append(row)
    summary = []
    for (dataset, selector, track, k), group in sorted(grouped.items()):
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary.append({
                "dataset": dataset,
                "selector": selector,
                "track": track,
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            })
    return summary


def run_anchor_discovery(
    split: str,
    output: Path,
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    max_hops: int = 8,
    hop_decay: float = 0.7,
    device: str = "cpu",
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise RuntimeError("no-oracle anchor development is restricted to train/dev")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive values")

    spec = load_holdout_spec()
    traces = int(spec["traces_per_seed"])
    dataset_names = tuple(str(name) for name in spec["datasets"])
    seeds = split_seeds(split)
    max_k = max(ks)
    output.mkdir(parents=True, exist_ok=True)

    base_embedder = SentenceTransformersEmbedder(
        ANCHOR_MODEL.model_name,
        revision=ANCHOR_MODEL.revision,
        device=device,
        normalize_embeddings=True,
        expected_dimensions=ANCHOR_MODEL.dimensions,
    )
    embedder = CachedEmbedder(base_embedder)
    lexical = RankBM25Retriever()

    anchor_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []

    for dataset_name in dataset_names:
        for seed in seeds:
            dataset = generate(dataset_name, seed, traces)
            retriever = CTRetriever(
                dataset.topology,
                embedder=embedder,
                lexical_retriever=lexical,
                hop_decay=hop_decay,
            )
            for query in dataset.queries:
                # Raw query text contains no opaque node/event IDs.
                if any(node_id in query.text for node_id in dataset.topology.nodes):
                    raise RuntimeError(f"query {query.id} leaks an opaque node id")
                selections = _rankings(retriever, query.text)
                oracle_hits = retriever.search(
                    query.text,
                    mode=query.mode,
                    anchor_ids=[query.anchor],
                    k=max_k,
                    max_hops=max_hops,
                    exhaustive=True,
                )
                oracle_ranking = [hit.node.id for hit in oracle_hits]

                for selector, ranking in selections.items():
                    candidates = ranking[:ANCHOR_K]
                    top1 = candidates[0]
                    top1_correct = top1 == query.anchor
                    topk_correct = query.anchor in candidates
                    anchor_rows.append({
                        "split": split,
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "mode": query.mode.value,
                        "selector": selector,
                        "gold_anchor": query.anchor,
                        "selected_anchor": top1,
                        "anchor_candidates": candidates,
                        "anchor_top1_correct": int(top1_correct),
                        "anchor_topk_correct": int(topk_correct),
                    })

                    discovered_hits = retriever.search(
                        query.text,
                        mode=query.mode,
                        anchor_ids=[top1],
                        k=max_k,
                        max_hops=max_hops,
                        exhaustive=True,
                    )
                    uncertain_hits = retriever.search(
                        query.text,
                        mode=query.mode,
                        anchor_ids=candidates,
                        k=max_k,
                        max_hops=max_hops,
                        exhaustive=True,
                    )
                    tracks = {
                        "oracle_anchor": oracle_ranking,
                        "discovered_top1": [hit.node.id for hit in discovered_hits],
                        "uncertain_top3": [hit.node.id for hit in uncertain_hits],
                    }
                    for track, retrieved in tracks.items():
                        for k in ks:
                            metrics = evaluate(retrieved[:k], query, dataset.topology, k)
                            error_class = "none"
                            if track != "oracle_anchor":
                                if not top1_correct and track == "discovered_top1":
                                    error_class = "anchor_selection"
                                elif metrics.get("causal_path_recall") is not None and metrics["causal_path_recall"] < 1.0:
                                    error_class = "traversal_or_reranking"
                            downstream_rows.append({
                                "split": split,
                                "dataset": dataset_name,
                                "seed": seed,
                                "query_id": query.id,
                                "mode": query.mode.value,
                                "selector": selector,
                                "track": track,
                                "k": k,
                                "anchor_top1_correct": int(top1_correct),
                                "anchor_top3_contains_gold": int(topk_correct),
                                "error_class": error_class,
                                **metrics,
                            })

            for ambiguous in _ambiguous_queries(dataset.topology):
                selections = _rankings(retriever, ambiguous["text"])
                gold = set(ambiguous["gold_anchor_ids"])
                for selector, ranking in selections.items():
                    ambiguous_rows.append({
                        "split": split,
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_text": ambiguous["text"],
                        "selector": selector,
                        "gold_anchor_count": len(gold),
                        "top1_matches_any_gold": int(ranking[0] in gold),
                        "top3_matches_any_gold": int(bool(set(ranking[:ANCHOR_K]) & gold)),
                    })

    manifest = {
        "schema_version": 1,
        "experiment": "no_oracle_anchor_discovery",
        "split": split,
        "final_test_executed": False,
        "dataset_sha256": split_dataset_sha256(split),
        "preregistration": preregistration_manifest(),
        "anchor_model": base_embedder.descriptor(),
        "lexical": lexical.descriptor(),
        "selectors": ["semantic", "lexical", "hybrid"],
        "anchor_k": ANCHOR_K,
        "tracks": ["oracle_anchor", "discovered_top1", "uncertain_top3"],
        "uncertainty_policy": "top-3 anchor candidates are all propagated into causal/topological expansion",
        "opaque_id_policy": "raw query text is rejected if it contains any generated node id",
    }
    report = {
        "manifest": manifest,
        "anchor_results": anchor_rows,
        "anchor_summary": _summarize_anchor(anchor_rows),
        "downstream_results": downstream_rows,
        "downstream_summary": _summarize_downstream(downstream_rows),
        "ambiguous_results": ambiguous_rows,
    }
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    (output / "results.json").write_text(_json(report), encoding="utf-8", newline="\n")
    _csv(output / "anchor-results.csv", [
        dict(row, anchor_candidates=json.dumps(row["anchor_candidates"])) for row in anchor_rows
    ])
    _csv(output / "anchor-summary.csv", report["anchor_summary"])
    _csv(output / "downstream-results.csv", downstream_rows)
    _csv(output / "downstream-summary.csv", report["downstream_summary"])
    _csv(output / "ambiguous-results.csv", ambiguous_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CT-RAG no-oracle anchor discovery")
    parser.add_argument("split", choices=("train", "dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_anchor_discovery(
        args.split,
        args.output or Path(f"anchor-discovery-results/{args.split}"),
        device=args.device,
    )
    print(
        f"Wrote {len(report['anchor_results'])} anchor-selection observations for {args.split}; "
        "oracle and discovered tracks remain separated"
    )


if __name__ == "__main__":
    main()
