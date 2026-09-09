from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ctrag.adapters import RankBM25Retriever, SentenceTransformersEmbedder
from ctrag.directional_retriever import CTRetriever
from ctrag.embedding import tokenize
from ctrag.models import Edge, EdgeKind, MemoryNode
from ctrag.topology import CausalTopology
from .datasets import Dataset, generate
from .holdout import load_holdout_spec, split_dataset_sha256, split_seeds
from .metrics import evaluate
from .protocol import preregistration_manifest

BASINRAG_REPOSITORY = "https://github.com/Basinfy/BasinRAG"
BASINRAG_COMMIT = "fb62771eda11f1a70d6e99ca7aa5e19b9825b829"
BASINRAG_VERSION = "1.0.3"
MATCHED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MATCHED_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MATCHED_MODEL_DIMENSIONS = 384
DEFAULT_CONTEXT_TOKEN_BUDGET = 64


@dataclass(frozen=True, slots=True)
class BasinRAGProtocol:
    upstream_repository: str = BASINRAG_REPOSITORY
    upstream_commit: str = BASINRAG_COMMIT
    upstream_version: str = BASINRAG_VERSION
    encoder_model: str = MATCHED_MODEL_NAME
    encoder_revision: str = MATCHED_MODEL_REVISION
    encoder_dimensions: int = MATCHED_MODEL_DIMENSIONS
    search_component: str = "BasinTopologyEngine + partition_into_basins + HybridSearch"
    reranker: str = "disabled for matched topology comparison"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _token_budget(ranking: list[str], topology: CausalTopology, budget: int, k: int) -> list[str]:
    selected: list[str] = []
    used = 0
    for node_id in ranking:
        if node_id not in topology.nodes:
            continue
        cost = max(1, len(tokenize(topology.nodes[node_id].text)))
        if selected and used + cost > budget:
            break
        if used + cost > budget:
            continue
        selected.append(node_id)
        used += cost
        if len(selected) >= k:
            break
    return selected


def _without_anchor(ranking: list[str], anchor: str) -> list[str]:
    return [node_id for node_id in ranking if node_id != anchor]


def _clone_without_causal_edges(dataset: Dataset) -> CausalTopology:
    topology = CausalTopology()
    for node in dataset.topology.nodes.values():
        topology.add_node(MemoryNode.from_dict(node.to_dict()))
    for source in sorted(dataset.topology.nodes):
        for edge in dataset.topology.outgoing(source):
            if edge.kind is EdgeKind.CAUSAL:
                continue
            topology.add_edge(Edge.from_dict(edge.to_dict()))
    return topology


def _basinrag_chunks(dataset: Dataset, embedder: SentenceTransformersEmbedder) -> list[dict[str, Any]]:
    grouped: dict[str, list[MemoryNode]] = defaultdict(list)
    for node in dataset.topology.nodes.values():
        execution = str(node.metadata.get("execution_id") or f"isolated:{node.id}")
        grouped[execution].append(node)

    chunks: list[dict[str, Any]] = []
    for execution, nodes in sorted(grouped.items()):
        ordered = sorted(nodes, key=lambda node: (node.timestamp, node.id))
        source = f"{dataset.name}:{dataset.seed}:{execution}"
        for chunk_index, node in enumerate(ordered):
            chunks.append({
                "id": node.id,
                "text": node.text,
                "embedding": np.asarray(embedder.embed(node.text), dtype=np.float32),
                "source": source,
                "chunk_index": chunk_index,
                "l1": "",
                "l2": "",
                "metadata": dict(node.metadata),
            })
    return chunks


def _build_upstream_basinrag(dataset: Dataset, embedder: SentenceTransformersEmbedder, storage_dir: str):
    try:
        from basinrag.core.topology import BasinTopologyEngine
        from basinrag.indexer.bm25 import BM25Index
        from basinrag.retriever.hybrid_search import HybridSearch
        from basinrag.retriever.local_search import TopologicalLocalSearch
    except ImportError as exc:  # pragma: no cover - exercised in science workflow
        raise RuntimeError(
            "BasinRAG reproduction requires the pinned upstream dependency; "
            f"install git+{BASINRAG_REPOSITORY}.git@{BASINRAG_COMMIT}"
        ) from exc

    chunks = _basinrag_chunks(dataset, embedder)
    engine = BasinTopologyEngine(storage_dir=storage_dir)
    engine.encoder_model = MATCHED_MODEL_NAME
    engine.build_graph(chunks)
    engine.partition_into_basins()
    engine.build_meta_basins()

    ids = [node_id for node_id in engine.graph.nodes]
    texts = [str(engine.graph.nodes[node_id].get("text", "")) for node_id in ids]
    bm25 = BM25Index()
    bm25.build(ids, texts)
    engine.bm25 = bm25
    local = TopologicalLocalSearch(engine)
    hybrid = HybridSearch(engine, local)
    return engine, hybrid


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = {"split", "dataset", "seed", "query_id", "mode", "arm", "k", "retrieved_ids"}
    metrics = [key for key in rows[0] if key not in identity]
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["mode"], row["arm"], row["k"])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, mode, arm, k), group in sorted(grouped.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary.append({
                "dataset": dataset,
                "mode": mode,
                "arm": arm,
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            })
    return summary


def _append(
    rows: list[dict[str, Any]],
    *,
    split: str,
    dataset: Dataset,
    query,
    arm: str,
    ranking: list[str],
    k: int,
) -> None:
    rows.append({
        "split": split,
        "dataset": dataset.name,
        "seed": dataset.seed,
        "query_id": query.id,
        "mode": query.mode.value,
        "arm": arm,
        "k": k,
        "retrieved_ids": ranking,
        **evaluate(ranking, query, dataset.topology, k),
    })


def run_basinrag_reproduction(
    split: str,
    output: Path,
    *,
    k: int = 3,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    max_hops: int = 8,
    hop_decay: float = 0.7,
    device: str = "cpu",
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise RuntimeError("BasinRAG reproduction is restricted to train/dev while final test is sealed")
    if k <= 0 or context_token_budget <= 0:
        raise ValueError("k and context_token_budget must be positive")

    spec = load_holdout_spec()
    seeds = split_seeds(split)
    traces = int(spec["traces_per_seed"])
    datasets = tuple(str(name) for name in spec["datasets"])
    output.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformersEmbedder(
        MATCHED_MODEL_NAME,
        revision=MATCHED_MODEL_REVISION,
        device=device,
        normalize_embeddings=True,
        expected_dimensions=MATCHED_MODEL_DIMENSIONS,
    )
    bm25 = RankBM25Retriever(k1=1.5, b=0.75, epsilon=0.25)
    rows: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []

    for dataset_name in datasets:
        for seed in seeds:
            dataset = generate(dataset_name, seed, traces)
            with tempfile.TemporaryDirectory(prefix="ctrag-basinrag-") as storage_dir:
                started = time.perf_counter()
                _engine, upstream_hybrid = _build_upstream_basinrag(dataset, embedder, storage_dir)
                costs.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "arm": "basinrag_upstream_hybrid",
                    "operation": "index",
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                })

                ct_full_dataset = generate(dataset_name, seed, traces)
                ct_full = CTRetriever(
                    ct_full_dataset.topology,
                    embedder=embedder,
                    lexical_retriever=bm25,
                    hop_decay=hop_decay,
                )
                no_causal_topology = _clone_without_causal_edges(dataset)
                ct_no_causal = CTRetriever(
                    no_causal_topology,
                    embedder=embedder,
                    lexical_retriever=bm25,
                    hop_decay=hop_decay,
                )

                for query in dataset.queries:
                    started = time.perf_counter()
                    query_embedding = np.asarray(embedder.embed(query.text), dtype=np.float32)
                    basin_hits = upstream_hybrid.search_nodes(
                        query.text,
                        query_embedding,
                        top_k=max(k * 4, k),
                    )
                    basin_ranking = _without_anchor([str(item["id"]) for item in basin_hits], query.anchor)
                    basin_ranking = _token_budget(basin_ranking, dataset.topology, context_token_budget, k)
                    costs.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "arm": "basinrag_upstream_hybrid",
                        "operation": "query",
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    })
                    _append(
                        rows,
                        split=split,
                        dataset=dataset,
                        query=query,
                        arm="basinrag_upstream_hybrid",
                        ranking=basin_ranking,
                        k=k,
                    )

                    started = time.perf_counter()
                    ct_hits = ct_full.search(
                        query.text,
                        mode=query.mode,
                        k=max(k * 4, k),
                        anchor_ids=None,
                        anchor_k=3,
                        max_hops=max_hops,
                        exhaustive=True,
                    )
                    ct_ranking = _without_anchor([hit.node.id for hit in ct_hits], query.anchor)
                    ct_ranking = _token_budget(ct_ranking, dataset.topology, context_token_budget, k)
                    costs.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "arm": "ctrag_full_no_oracle",
                        "operation": "query",
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    })
                    _append(
                        rows,
                        split=split,
                        dataset=dataset,
                        query=query,
                        arm="ctrag_full_no_oracle",
                        ranking=ct_ranking,
                        k=k,
                    )

                    started = time.perf_counter()
                    nc_hits = ct_no_causal.search(
                        query.text,
                        mode=query.mode,
                        k=max(k * 4, k),
                        anchor_ids=None,
                        anchor_k=3,
                        max_hops=max_hops,
                        exhaustive=True,
                    )
                    nc_ranking = _without_anchor([hit.node.id for hit in nc_hits], query.anchor)
                    nc_ranking = _token_budget(nc_ranking, dataset.topology, context_token_budget, k)
                    costs.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "query_id": query.id,
                        "arm": "ctrag_no_causal_metadata",
                        "operation": "query",
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    })
                    _append(
                        rows,
                        split=split,
                        dataset=dataset,
                        query=query,
                        arm="ctrag_no_causal_metadata",
                        ranking=nc_ranking,
                        k=k,
                    )

                    oracle_hits = ct_full.search(
                        query.text,
                        mode=query.mode,
                        k=max(k * 4, k),
                        anchor_ids=[query.anchor],
                        max_hops=max_hops,
                        exhaustive=True,
                    )
                    oracle_ranking = _without_anchor([hit.node.id for hit in oracle_hits], query.anchor)
                    oracle_ranking = _token_budget(oracle_ranking, dataset.topology, context_token_budget, k)
                    _append(
                        rows,
                        split=split,
                        dataset=dataset,
                        query=query,
                        arm="ctrag_oracle_diagnostic",
                        ranking=oracle_ranking,
                        k=k,
                    )

    summary = _aggregate(rows)
    cost_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in costs:
        cost_groups[(row["arm"], row["operation"])].append(float(row["elapsed_ms"]))
    cost_summary = [
        {
            "arm": arm,
            "operation": operation,
            "n": len(values),
            "mean_ms": statistics.mean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for (arm, operation), values in sorted(cost_groups.items())
    ]

    manifest = {
        "schema_version": 1,
        "experiment": "basinrag_upstream_reproduction_matched_comparison",
        "split": split,
        "final_test_executed": False,
        "dataset_sha256": split_dataset_sha256(split),
        "preregistration": preregistration_manifest(),
        "basinrag": asdict(BasinRAGProtocol()),
        "matched_protocol": {
            "corpus": "identical synthetic MemoryNode text corpus",
            "query": "identical raw query text for end-to-end arms",
            "k": k,
            "context_token_budget": context_token_budget,
            "dense_model": MATCHED_MODEL_NAME,
            "dense_revision": MATCHED_MODEL_REVISION,
            "dense_dimensions": MATCHED_MODEL_DIMENSIONS,
            "ctrag_anchor_policy": "discovered top-3 for end-to-end; oracle diagnostic reported separately",
            "basinrag_causal_metadata": "none; only source/chunk order, text, embedding and upstream virtual edges",
            "ctrag_no_causal_ablation": "all explicit CAUSAL edges removed before retrieval",
        },
        "notes": [
            "This uses BasinRAG upstream topology and HybridSearch directly at the pinned commit.",
            "The cross-encoder reranker is disabled in both matched arms to isolate retrieval/topology rather than reranking model quality.",
            "BasinRAG receives execution grouping only as document/source sequence; branch causation is not encoded.",
            "The oracle CT-RAG arm is diagnostic only and must not be presented as end-to-end performance.",
        ],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    report = {
        "manifest": manifest,
        "results": rows,
        "summary": summary,
        "costs": costs,
        "cost_summary": cost_summary,
    }
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    (output / "results.json").write_text(_json(report), encoding="utf-8", newline="\n")
    _csv(output / "results.csv", [
        dict(row, retrieved_ids=json.dumps(row["retrieved_ids"])) for row in rows
    ])
    _csv(output / "summary.csv", summary)
    _csv(output / "costs.csv", costs)
    _csv(output / "cost-summary.csv", cost_summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce pinned upstream BasinRAG on CT-RAG train/dev")
    parser.add_argument("split", choices=("train", "dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--context-token-budget", type=int, default=DEFAULT_CONTEXT_TOKEN_BUDGET)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--hop-decay", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output or Path(f"basinrag-reproduction-results/{args.split}")
    report = run_basinrag_reproduction(
        args.split,
        output,
        k=args.k,
        context_token_budget=args.context_token_budget,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        device=args.device,
    )
    print(f"Wrote {len(report['results'])} matched BasinRAG observations for {args.split}")


if __name__ == "__main__":
    main()
