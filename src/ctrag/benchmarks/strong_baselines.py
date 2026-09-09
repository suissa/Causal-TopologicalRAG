from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ctrag.adapters import (
    EmbeddingProvider,
    RankBM25Retriever,
    SentenceTransformersEmbedder,
    reciprocal_rank_fusion,
)
from ctrag.directional_retriever import CTRetriever
from ctrag.models import QueryMode
from .datasets import generate
from .holdout import load_holdout_spec, split_dataset_sha256, split_seeds
from .metrics import evaluate
from .protocol import preregistration_manifest


@dataclass(frozen=True, slots=True)
class DenseModelSpec:
    key: str
    model_name: str
    revision: str
    dimensions: int
    language: str = "en"
    normalize_embeddings: bool = True


STRONG_DENSE_MODELS: tuple[DenseModelSpec, ...] = (
    DenseModelSpec(
        key="minilm_l6_v2",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        dimensions=384,
    ),
    DenseModelSpec(
        key="mpnet_base_v2",
        model_name="sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        dimensions=768,
    ),
)


class CachedEmbedder:
    """Cache exact text embeddings so every arm sees identical dense vectors."""

    def __init__(self, delegate: EmbeddingProvider) -> None:
        self.delegate = delegate
        self.cache: dict[str, tuple[float, ...]] = {}

    def embed(self, text: str) -> tuple[float, ...]:
        if text not in self.cache:
            self.cache[text] = self.delegate.embed(text)
        return self.cache[text]


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _max_rss_kb() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows compatibility
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes while Linux reports KiB.
    return value // 1024 if platform.system() == "Darwin" else value


def _without_anchor(ranking: list[str], anchor: str) -> list[str]:
    return [node_id for node_id in ranking if node_id != anchor]


def _quality_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = {"split", "dataset", "seed", "query_id", "mode", "baseline", "k", "retrieved_ids"}
    metric_names = [key for key in rows[0] if key not in identity]
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["mode"], row["baseline"], row["k"])].append(row)

    summary: list[dict[str, Any]] = []
    for (dataset, mode, baseline, k), group in sorted(grouped.items()):
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary.append({
                "dataset": dataset,
                "mode": mode,
                "baseline": baseline,
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            })
    return summary


def _cost_summary(cost_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cost_rows:
        grouped[(row["baseline"], row["operation"])].append(row)
    result: list[dict[str, Any]] = []
    for (baseline, operation), group in sorted(grouped.items()):
        values = [float(row["elapsed_ms"]) for row in group]
        result.append({
            "baseline": baseline,
            "operation": operation,
            "n": len(values),
            "mean_ms": statistics.mean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "max_rss_kb": max((row["max_rss_kb"] or 0) for row in group) or None,
        })
    return result


def _append_quality_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    dataset_name: str,
    seed: int,
    query,
    baseline: str,
    ranking: list[str],
    topology,
    ks: tuple[int, ...],
) -> None:
    for k in ks:
        ranked = ranking[:k]
        rows.append({
            "split": split,
            "dataset": dataset_name,
            "seed": seed,
            "query_id": query.id,
            "mode": query.mode.value,
            "baseline": baseline,
            "k": k,
            "retrieved_ids": ranked,
            **evaluate(ranked, query, topology, k),
        })


def run_strong_baselines(
    split: str,
    output: Path,
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    max_hops: int = 8,
    hop_decay: float = 0.7,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run strong retrieval comparisons on train/dev only.

    The pristine final-test split is deliberately unavailable here. Final
    evaluation is controlled exclusively by the sealed holdout command.
    """
    if split not in {"train", "dev"}:
        raise RuntimeError("strong-baseline tuning/evaluation is restricted to train/dev")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive values")

    holdout_spec = load_holdout_spec()
    seeds = split_seeds(split)
    traces = int(holdout_spec["traces_per_seed"])
    dataset_names = tuple(str(name) for name in holdout_spec["datasets"])
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    model_manifests: list[dict[str, Any]] = []
    bm25 = RankBM25Retriever(k1=1.5, b=0.75, epsilon=0.25)
    max_k = max(ks)

    # BM25 is model-independent, so evaluate it exactly once per query.
    for dataset_name in dataset_names:
        for seed in seeds:
            dataset = generate(dataset_name, seed, traces)
            documents = {node.id: node.text for node in dataset.topology.nodes.values()}
            for query in dataset.queries:
                started = time.perf_counter()
                scored = bm25.score(query.text, documents)
                ranking = sorted(scored, key=lambda node_id: (-scored[node_id], node_id))
                ranking = _without_anchor(ranking, query.anchor)
                elapsed = (time.perf_counter() - started) * 1000.0
                cost_rows.append({
                    "baseline": "bm25_rank_bm25",
                    "operation": "query",
                    "dataset": dataset_name,
                    "seed": seed,
                    "query_id": query.id,
                    "elapsed_ms": elapsed,
                    "max_rss_kb": _max_rss_kb(),
                })
                _append_quality_rows(
                    rows,
                    split=split,
                    dataset_name=dataset_name,
                    seed=seed,
                    query=query,
                    baseline="bm25_rank_bm25",
                    ranking=ranking,
                    topology=dataset.topology,
                    ks=ks,
                )

    for model_spec in STRONG_DENSE_MODELS:
        load_started = time.perf_counter()
        delegate = SentenceTransformersEmbedder(
            model_spec.model_name,
            revision=model_spec.revision,
            device=device,
            normalize_embeddings=model_spec.normalize_embeddings,
            expected_dimensions=model_spec.dimensions,
        )
        embedder = CachedEmbedder(delegate)
        load_ms = (time.perf_counter() - load_started) * 1000.0
        descriptor = delegate.descriptor()
        descriptor["key"] = model_spec.key
        descriptor["language"] = model_spec.language
        descriptor["model_load_ms"] = load_ms
        descriptor["max_rss_kb_after_load"] = _max_rss_kb()
        model_manifests.append(descriptor)
        cost_rows.append({
            "baseline": f"dense_{model_spec.key}",
            "operation": "model_load",
            "dataset": "*",
            "seed": -1,
            "query_id": "*",
            "elapsed_ms": load_ms,
            "max_rss_kb": _max_rss_kb(),
        })

        for dataset_name in dataset_names:
            for seed in seeds:
                # Regenerate per model so node.embedding cannot leak across model dimensions.
                dataset = generate(dataset_name, seed, traces)
                index_started = time.perf_counter()
                retriever = CTRetriever(
                    dataset.topology,
                    embedder=embedder,
                    lexical_retriever=bm25,
                    hop_decay=hop_decay,
                )
                index_ms = (time.perf_counter() - index_started) * 1000.0
                cost_rows.append({
                    "baseline": f"dense_{model_spec.key}",
                    "operation": "index",
                    "dataset": dataset_name,
                    "seed": seed,
                    "query_id": "*",
                    "elapsed_ms": index_ms,
                    "max_rss_kb": _max_rss_kb(),
                })

                for query in dataset.queries:
                    started = time.perf_counter()
                    dense_ranking = [
                        node_id for node_id, _ in retriever.rank_dense(
                            query.text, k=len(dataset.topology.nodes)
                        )
                    ]
                    dense_ranking = _without_anchor(dense_ranking, query.anchor)
                    dense_ms = (time.perf_counter() - started) * 1000.0
                    dense_name = f"dense_{model_spec.key}"
                    cost_rows.append({
                        "baseline": dense_name,
                        "operation": "query",
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "elapsed_ms": dense_ms,
                        "max_rss_kb": _max_rss_kb(),
                    })
                    _append_quality_rows(
                        rows,
                        split=split,
                        dataset_name=dataset_name,
                        seed=seed,
                        query=query,
                        baseline=dense_name,
                        ranking=dense_ranking,
                        topology=dataset.topology,
                        ks=ks,
                    )

                    started = time.perf_counter()
                    lexical_ranking = [
                        node_id for node_id, _ in retriever.rank_lexical(
                            query.text, k=len(dataset.topology.nodes)
                        )
                    ]
                    lexical_ranking = _without_anchor(lexical_ranking, query.anchor)
                    fused = [
                        node_id for node_id, _ in reciprocal_rank_fusion(
                            [dense_ranking, lexical_ranking], rank_constant=60
                        )
                    ]
                    hybrid_ms = (time.perf_counter() - started) * 1000.0
                    hybrid_name = f"hybrid_{model_spec.key}_bm25"
                    cost_rows.append({
                        "baseline": hybrid_name,
                        "operation": "query",
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "elapsed_ms": hybrid_ms,
                        "max_rss_kb": _max_rss_kb(),
                    })
                    _append_quality_rows(
                        rows,
                        split=split,
                        dataset_name=dataset_name,
                        seed=seed,
                        query=query,
                        baseline=hybrid_name,
                        ranking=fused,
                        topology=dataset.topology,
                        ks=ks,
                    )

                    started = time.perf_counter()
                    ct_hits = retriever.search(
                        query.text,
                        mode=query.mode,
                        anchor_ids=[query.anchor],
                        k=max_k,
                        max_hops=max_hops,
                        exhaustive=True,
                    )
                    ct_ranking = _without_anchor([hit.node.id for hit in ct_hits], query.anchor)
                    ct_ms = (time.perf_counter() - started) * 1000.0
                    ct_name = f"full_ctrag_{model_spec.key}_bm25"
                    cost_rows.append({
                        "baseline": ct_name,
                        "operation": "query",
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "elapsed_ms": ct_ms,
                        "max_rss_kb": _max_rss_kb(),
                    })
                    _append_quality_rows(
                        rows,
                        split=split,
                        dataset_name=dataset_name,
                        seed=seed,
                        query=query,
                        baseline=ct_name,
                        ranking=ct_ranking,
                        topology=dataset.topology,
                        ks=ks,
                    )

    quality_summary = _quality_summary(rows)
    cost_summary = _cost_summary(cost_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "strong_retrieval_baselines",
        "split": split,
        "final_test_executed": False,
        "dataset_sha256": split_dataset_sha256(split),
        "preregistration": preregistration_manifest(),
        "ks": list(ks),
        "max_hops": max_hops,
        "hop_decay": hop_decay,
        "anchor_policy": "same oracle anchor visible to all arms; known anchor excluded from ranking",
        "candidate_policy": "same exhaustive corpus for all arms",
        "bm25": bm25.descriptor(),
        "dense_models": model_manifests,
        "fusion": {"name": "reciprocal_rank_fusion", "rank_constant": 60},
        "proxy_baselines_are_competitive": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    report = {
        "manifest": manifest,
        "results": rows,
        "summary": quality_summary,
        "costs": cost_rows,
        "cost_summary": cost_summary,
    }

    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    (output / "results.json").write_text(_json(report), encoding="utf-8", newline="\n")
    _csv(output / "results.csv", [
        dict(row, retrieved_ids=json.dumps(row["retrieved_ids"])) for row in rows
    ])
    _csv(output / "summary.csv", quality_summary)
    _csv(output / "costs.csv", cost_rows)
    _csv(output / "cost-summary.csv", cost_summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CT-RAG strong retrieval baselines on train/dev")
    parser.add_argument("split", choices=("train", "dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--hop-decay", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_strong_baselines(
        args.split,
        args.output or Path(f"strong-baseline-results/{args.split}"),
        ks=tuple(args.ks),
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        device=args.device,
    )
    print(
        f"Wrote {len(report['results'])} strong-baseline observations for {args.split}; "
        "final test remained sealed"
    )


if __name__ == "__main__":
    main()
