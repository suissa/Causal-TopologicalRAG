from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ctrag.directional_retriever import CTRetriever
from ctrag.embedding import HashingEmbedder
from ctrag.models import QueryMode, RetrievalWeights
from .datasets import GENERATOR_VERSION, generate
from .metrics import evaluate
from .protocol import preregistration_manifest

BASELINES = {
    "lexical_only": RetrievalWeights(0, 1, 0, 0, 0, 0),
    "dense_only": RetrievalWeights(1, 0, 0, 0, 0, 0),
    "dense_lexical": RetrievalWeights(.5, .5, 0, 0, 0, 0),
    "graph_topology_only": RetrievalWeights(0, 0, 0, 1, 0, 0),
    "dense_causal": RetrievalWeights(.5, 0, .5, 0, 0, 0),
    "dense_topological": RetrievalWeights(.5, 0, 0, .5, 0, 0),
    "full_ctrag": None,
}


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (7, 42, 2024)
    ks: tuple[int, ...] = (1, 3, 5, 10)
    traces: int = 4
    dimensions: int = 256
    max_hops: int = 8
    hop_decay: float = .7

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("provide unique seeds")
        if not self.ks or any(k < 1 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError("provide unique positive K values")
        if min(self.traces, self.dimensions, self.max_hops) < 1:
            raise ValueError("traces, dimensions and max_hops must be positive")
        if not 0 < self.hop_decay < float("inf"):
            raise ValueError("hop_decay must be finite and positive")


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: Config, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifests, rows = [], []
    for name in ("failure_recovery", "branching"):
        for seed in config.seeds:
            dataset = generate(name, seed, config.traces)
            manifests.append(dataset.manifest())
            retriever = CTRetriever(dataset.topology, embedder=HashingEmbedder(config.dimensions),
                                    hop_decay=config.hop_decay)
            for query in dataset.queries:
                for baseline, weights in BASELINES.items():
                    # The same known state and entire corpus are given to every arm.
                    hits = retriever.search(query.text, mode=query.mode, anchor_ids=[query.anchor],
                                            k=max(config.ks), max_hops=config.max_hops,
                                            weights=weights, exhaustive=True)
                    for k in config.ks:
                        ranked = [hit.node.id for hit in hits[:k]]
                        rows.append(dict(dataset=name, seed=seed, query_id=query.id,
                                         mode=query.mode.value, baseline=baseline, k=k,
                                         retrieved_ids=ranked,
                                         **evaluate(ranked, query, dataset.topology, k)))
    grouped = defaultdict(list)
    fields = ("dataset", "mode", "baseline", "k")
    metrics = [key for key in rows[0] if key not in (*fields, "seed", "query_id", "retrieved_ids")]
    for row in rows:
        grouped[tuple(row[key] for key in fields)].append(row)
    summary, table = [], []
    for key, group in sorted(grouped.items()):
        table_row = dict(zip(fields, key))
        for metric in metrics:
            values = [row[metric] for row in group if row[metric] is not None]
            mean = statistics.mean(values) if values else None
            summary.append(dict(zip(fields, key), metric=metric, n=len(values), mean=mean,
                                std=statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)))
            table_row[metric] = mean
        table.append(table_row)
    dataset_json = _json(manifests)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {path.relative_to(source_root).as_posix(): hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for path in sorted(source_root.rglob("*.py"))}
    manifest = dict(
        schema_version=1, generator_version=GENERATOR_VERSION, **asdict(config),
        preregistration=preregistration_manifest(),
        datasets_sha256=hashlib.sha256(dataset_json.encode("utf-8")).hexdigest(),
        embedding={"name": "HashingEmbedder", "dimensions": config.dimensions,
                   "learned_dense_model": False},
        candidate_policy="exhaustive", anchor_policy="explicit_known_state_excluded_from_ranking",
        tie_break="score_desc_node_id_asc", token_counter="ctrag.embedding.tokenize",
        basin_max_hops=8,
        weights={name: {mode.value: asdict(weights or RetrievalWeights.for_mode(mode))
                        for mode in QueryMode} for name, weights in BASELINES.items()},
        aggregation="macro over applicable query/seed observations; sample standard deviation",
        python_version=platform.python_version(),
        source_sha256=source_hashes,
    )
    # Keep the return value and its JSON representation identical (including arrays).
    manifest = json.loads(_json(manifest))
    report = dict(config=manifest, results=rows, summary=summary)
    (output / "datasets.json").write_text(dataset_json, encoding="utf-8", newline="\n")
    (output / "config.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    (output / "results.json").write_text(_json(report), encoding="utf-8", newline="\n")
    _csv(output / "results.csv", [dict(row, retrieved_ids=json.dumps(row["retrieved_ids"])) for row in rows])
    _csv(output / "summary.csv", summary)
    _csv(output / "table.csv", table)
    return report
