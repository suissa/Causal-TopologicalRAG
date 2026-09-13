from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from ctrag.directional_retriever import CTRetriever
from ctrag.models import Edge, EdgeKind, MemoryNode, QueryMode
from ctrag.topology import CausalTopology
from .datasets import Dataset, Query
from .metrics import evaluate

DEFAULT_FIXTURE = Path("research/external/github-actions-v1.json")


@dataclass(frozen=True, slots=True)
class ExternalSource:
    dataset_id: str
    repository: str
    run_id: int
    job_id: int
    run_url: str
    conclusion: str
    independent_public: bool
    steps: tuple[tuple[int, str, str], ...]


def load_sources(path: Path = DEFAULT_FIXTURE) -> tuple[ExternalSource, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported external fixture schema")
    policy = raw.get("transformation_policy", {})
    if policy.get("step_sequence") != "temporal_only":
        raise ValueError("step sequence must remain temporal_only")
    result: list[ExternalSource] = []
    for item in raw["sources"]:
        steps = tuple((int(n), str(name), str(status)) for n, name, status in item["steps"])
        result.append(ExternalSource(
            dataset_id=str(item["dataset_id"]), repository=str(item["repository"]),
            run_id=int(item["run_id"]), job_id=int(item["job_id"]), run_url=str(item["run_url"]),
            conclusion=str(item["conclusion"]), independent_public=bool(item["independent_public"]), steps=steps,
        ))
    return tuple(result)


def source_to_dataset(source: ExternalSource) -> Dataset:
    topology = CausalTopology()
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=source.run_id % 100000)
    node_ids: list[str] = []
    for index, (number, name, status) in enumerate(source.steps):
        node_id = f"gha:{source.repository}:{source.run_id}:{source.job_id}:{number}"
        node_ids.append(node_id)
        topology.add_node(MemoryNode(
            id=node_id,
            text=name,
            timestamp=epoch + timedelta(seconds=index),
            metadata={
                "execution_id": f"gha:{source.repository}:{source.run_id}:{source.job_id}",
                "repository": source.repository,
                "run_id": source.run_id,
                "job_id": source.job_id,
                "step_number": number,
                "status": status,
                "source_url": source.run_url,
                "provenance": "github_actions_public_api_snapshot",
                "causality": "unknown",
            },
        ))
    for left, right in zip(node_ids, node_ids[1:]):
        topology.add_edge(Edge(left, right, EdgeKind.TEMPORAL))

    queries: list[Query] = []
    for index in range(len(node_ids) - 1):
        anchor, target = node_ids[index], node_ids[index + 1]
        queries.append(Query(
            id=f"{source.dataset_id}:next:{index}",
            text=topology.nodes[anchor].text,
            mode=QueryMode.WHAT_NEXT,
            anchor=anchor,
            relevance={target: 2},
            causal_nodes=[], causal_paths=[], causal_distances={},
            trajectory=[anchor, target], basin_nodes=node_ids, recovery_nodes=None,
        ))

    failed = [i for i, (_, _, status) in enumerate(source.steps) if status == "failure"]
    for index in failed:
        if index > 0:
            anchor, predecessor = node_ids[index], node_ids[index - 1]
            queries.append(Query(
                id=f"{source.dataset_id}:pre-failure:{index}",
                text=f"last observed step before {topology.nodes[anchor].text}",
                mode=QueryMode.WHY,
                anchor=anchor,
                relevance={predecessor: 2},
                causal_nodes=[], causal_paths=[], causal_distances={},
                trajectory=[predecessor, anchor], basin_nodes=node_ids, recovery_nodes=None,
            ))
        recovery = next((j for j in range(index + 1, len(source.steps)) if source.steps[j][2] == "success"), None)
        if recovery is not None:
            anchor, target = node_ids[index], node_ids[recovery]
            queries.append(Query(
                id=f"{source.dataset_id}:observed-post-failure:{index}",
                text="observed successful cleanup or continuation after failure",
                mode=QueryMode.RECOVERY,
                anchor=anchor,
                relevance={target: 2},
                causal_nodes=[], causal_paths=[], causal_distances={},
                trajectory=[anchor, target], basin_nodes=node_ids, recovery_nodes=[target],
            ))
    return Dataset(source.dataset_id, source.run_id, topology, queries)


def evaluate_source(source: ExternalSource, *, k: int = 3) -> dict:
    dataset = source_to_dataset(source)
    retriever = CTRetriever(dataset.topology)
    rows = []
    for query in dataset.queries:
        arms = {
            "dense": [node for node, _ in retriever.rank_dense(query.text, k=k + 1)],
            "lexical": [node for node, _ in retriever.rank_lexical(query.text, k=k + 1)],
            "hybrid": [node for node, _ in retriever.rank_hybrid_rrf(query.text, k=k + 1)],
            "ctrag_oracle_anchor": [hit.node.id for hit in retriever.search(
                query.text, mode=query.mode, k=k + 1, anchor_ids=[query.anchor], exhaustive=True
            )],
        }
        for arm, ranking in arms.items():
            ranking = [node for node in ranking if node != query.anchor][:k]
            rows.append({"dataset": source.dataset_id, "query_id": query.id, "mode": query.mode.value,
                         "arm": arm, **evaluate(ranking, query, dataset.topology, k)})
    summary = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            metric: mean(float(row[metric]) for row in arm_rows if row[metric] is not None)
            for metric in ("recall_at_k", "mrr", "ndcg", "trajectory_reconstruction_accuracy")
            if any(row[metric] is not None for row in arm_rows)
        }
    return {"dataset": source.dataset_id, "repository": source.repository,
            "independent_public": source.independent_public, "queries": len(dataset.queries),
            "causal_edges": 0, "summary": summary, "rows": rows}


def run(output: Path, fixture: Path = DEFAULT_FIXTURE, *, k: int = 3) -> dict:
    sources = load_sources(fixture)
    results = [evaluate_source(source, k=k) for source in sources]
    report = {
        "schema_version": 1,
        "fixture": str(fixture),
        "final_holdout_executed": False,
        "causal_claim_policy": "No CAUSAL edges are created from step order. WHY/RECOVERY labels are observational only.",
        "datasets": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CT-RAG on versioned external GitHub Actions traces")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=Path("external-validity-results"))
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    report = run(args.output, args.fixture, k=args.k)
    print(f"Wrote external validity results for {len(report['datasets'])} datasets")


if __name__ == "__main__":
    main()
