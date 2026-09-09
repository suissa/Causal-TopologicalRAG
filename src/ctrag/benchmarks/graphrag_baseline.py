from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctrag.adapters import RankBM25Retriever, SentenceTransformersEmbedder, reciprocal_rank_fusion
from ctrag.directional_retriever import CTRetriever
from ctrag.embedding import cosine_similarity
from .datasets import generate
from .holdout import load_holdout_spec, split_dataset_sha256, split_seeds
from .metrics import evaluate
from .protocol import preregistration_manifest
from .strong_baselines import CachedEmbedder, STRONG_DENSE_MODELS


@dataclass(frozen=True, slots=True)
class RelationGraphConfig:
    neighbors: int = 3
    max_hops: int = 3

    def __post_init__(self) -> None:
        if self.neighbors < 1 or self.max_hops < 1:
            raise ValueError("neighbors and max_hops must be positive")


class SemanticRelationGraph:
    """Text/embedding-only relation graph; execution-causal edges are never read."""

    def __init__(self, retriever: CTRetriever, *, neighbors: int = 3) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.neighbors = neighbors
        self.adjacency: dict[str, set[str]] = {node_id: set() for node_id in retriever.topology.nodes}
        ids = sorted(retriever.topology.nodes)
        # CTRetriever has already materialized embeddings for every node.
        for source in ids:
            source_embedding = retriever.topology.nodes[source].embedding or ()
            candidates: list[tuple[float, str]] = []
            for target in ids:
                if target == source:
                    continue
                target_embedding = retriever.topology.nodes[target].embedding or ()
                similarity = cosine_similarity(source_embedding, target_embedding)
                candidates.append((similarity, target))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, target in candidates[:neighbors]:
                self.adjacency[source].add(target)
                self.adjacency[target].add(source)

    def distances(self, anchors: list[str], *, max_hops: int) -> dict[str, int]:
        missing = [anchor for anchor in anchors if anchor not in self.adjacency]
        if missing:
            raise KeyError(f"unknown graph anchors: {missing}")
        distances = {anchor: 0 for anchor in anchors}
        queue = deque(anchors)
        while queue:
            current = queue.popleft()
            hops = distances[current]
            if hops >= max_hops:
                continue
            for neighbor in sorted(self.adjacency[current]):
                if neighbor not in distances:
                    distances[neighbor] = hops + 1
                    queue.append(neighbor)
        return distances

    def rank(self, anchors: list[str], *, max_hops: int) -> list[str]:
        distances = self.distances(anchors, max_hops=max_hops)
        anchor_set = set(anchors)
        return [
            node_id
            for node_id, _ in sorted(distances.items(), key=lambda item: (item[1], item[0]))
            if node_id not in anchor_set
        ]


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = {"split", "dataset", "seed", "query_id", "mode", "model", "system", "k", "retrieved_ids"}
    metrics = [key for key in rows[0] if key not in identity]
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["mode"], row["model"], row["system"], row["k"])].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, mode, model, system, k), group in sorted(grouped.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            result.append({
                "dataset": dataset,
                "mode": mode,
                "model": model,
                "system": system,
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            })
    return result


def _append_quality(rows, *, split, dataset_name, seed, query, model, system, ranking, topology, ks):
    for k in ks:
        selected = ranking[:k]
        rows.append({
            "split": split,
            "dataset": dataset_name,
            "seed": seed,
            "query_id": query.id,
            "mode": query.mode.value,
            "model": model,
            "system": system,
            "k": k,
            "retrieved_ids": selected,
            **evaluate(selected, query, topology, k),
        })


def run_graphrag_comparison(
    split: str,
    output: Path,
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    graph_config: RelationGraphConfig = RelationGraphConfig(),
    device: str = "cpu",
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise RuntimeError("GraphRAG comparison is restricted to train/dev")
    spec = load_holdout_spec()
    traces = int(spec["traces_per_seed"])
    seeds = split_seeds(split)
    dataset_names = tuple(str(name) for name in spec["datasets"])
    output.mkdir(parents=True, exist_ok=True)
    max_k = max(ks)

    rows: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    lexical = RankBM25Retriever()

    for model_spec in STRONG_DENSE_MODELS:
        started = time.perf_counter()
        delegate = SentenceTransformersEmbedder(
            model_spec.model_name,
            revision=model_spec.revision,
            device=device,
            normalize_embeddings=True,
            expected_dimensions=model_spec.dimensions,
        )
        embedder = CachedEmbedder(delegate)
        model_manifest.append(delegate.descriptor())
        costs.append({
            "model": model_spec.key,
            "system": "shared",
            "operation": "model_load",
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        })

        for dataset_name in dataset_names:
            for seed in seeds:
                dataset = generate(dataset_name, seed, traces)
                started = time.perf_counter()
                retriever = CTRetriever(
                    dataset.topology,
                    embedder=embedder,
                    lexical_retriever=lexical,
                )
                graph = SemanticRelationGraph(retriever, neighbors=graph_config.neighbors)
                costs.append({
                    "model": model_spec.key,
                    "system": "semantic_graphrag",
                    "operation": "index",
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                })

                for query in dataset.queries:
                    size = len(dataset.topology.nodes)
                    embedder.invalidate(query.text)
                    started = time.perf_counter()
                    dense = [node_id for node_id, _ in retriever.rank_dense(query.text, k=size)]
                    lexical_rank = [node_id for node_id, _ in retriever.rank_lexical(query.text, k=size)]
                    # Oracle anchor is used here to isolate graph-structure value; #17 reports no-oracle separately.
                    relation_rank = graph.rank([query.anchor], max_hops=graph_config.max_hops)
                    graph_rank = [
                        node_id for node_id, _ in reciprocal_rank_fusion(
                            [dense, lexical_rank, relation_rank], rank_constant=60
                        )
                        if node_id != query.anchor
                    ]
                    graph_ms = (time.perf_counter() - started) * 1000.0
                    costs.append({
                        "model": model_spec.key,
                        "system": "semantic_graphrag",
                        "operation": "query_cold",
                        "elapsed_ms": graph_ms,
                    })
                    _append_quality(
                        rows,
                        split=split,
                        dataset_name=dataset_name,
                        seed=seed,
                        query=query,
                        model=model_spec.key,
                        system="semantic_graphrag",
                        ranking=graph_rank,
                        topology=dataset.topology,
                        ks=ks,
                    )

                    embedder.invalidate(query.text)
                    started = time.perf_counter()
                    ct_hits = retriever.search(
                        query.text,
                        mode=query.mode,
                        anchor_ids=[query.anchor],
                        k=max_k,
                        max_hops=8,
                        exhaustive=True,
                    )
                    ct_rank = [hit.node.id for hit in ct_hits if hit.node.id != query.anchor]
                    costs.append({
                        "model": model_spec.key,
                        "system": "full_ctrag",
                        "operation": "query_cold",
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    })
                    _append_quality(
                        rows,
                        split=split,
                        dataset_name=dataset_name,
                        seed=seed,
                        query=query,
                        model=model_spec.key,
                        system="full_ctrag",
                        ranking=ct_rank,
                        topology=dataset.topology,
                        ks=ks,
                    )

    manifest = {
        "schema_version": 1,
        "experiment": "matched_semantic_graphrag_comparison",
        "split": split,
        "final_test_executed": False,
        "dataset_sha256": split_dataset_sha256(split),
        "preregistration": preregistration_manifest(),
        "models": model_manifest,
        "lexical": lexical.descriptor(),
        "graph": {
            "type": "undirected semantic kNN relation graph",
            "neighbors": graph_config.neighbors,
            "max_hops": graph_config.max_hops,
            "construction_inputs": ["node text", "learned dense embedding"],
            "forbidden_inputs": ["causal edges", "causation ids", "execution topology", "gold relevance labels"],
        },
        "matching": {
            "same_corpus": True,
            "same_embedding_model": True,
            "same_lexical_model": True,
            "same_oracle_anchor": True,
            "same_retrieval_unit": "event node",
            "same_k_values": list(ks),
        },
        "claim_scope": "GraphRAG-style semantic relation graph baseline; not Microsoft GraphRAG implementation",
    }
    report = {
        "manifest": manifest,
        "results": rows,
        "summary": _summary(rows),
        "costs": costs,
    }
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    (output / "results.json").write_text(_json(report), encoding="utf-8", newline="\n")
    _csv(output / "results.csv", [dict(row, retrieved_ids=json.dumps(row["retrieved_ids"])) for row in rows])
    _csv(output / "summary.csv", report["summary"])
    _csv(output / "costs.csv", costs)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched GraphRAG-style vs CT-RAG comparison")
    parser.add_argument("split", choices=("train", "dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--graph-hops", type=int, default=3)
    args = parser.parse_args()
    report = run_graphrag_comparison(
        args.split,
        args.output or Path(f"graphrag-results/{args.split}"),
        graph_config=RelationGraphConfig(args.neighbors, args.graph_hops),
        device=args.device,
    )
    print(f"Wrote {len(report['results'])} matched GraphRAG comparison observations for {args.split}")


if __name__ == "__main__":
    main()
