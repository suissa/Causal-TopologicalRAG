from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ctrag.benchmarks.basinrag_reproduction import (
    BASINRAG_COMMIT,
    BASINRAG_REPOSITORY,
    BASINRAG_VERSION,
    BasinRAGProtocol,
    _clone_without_causal_edges,
    _csv,
    run_basinrag_reproduction,
)
from ctrag.benchmarks.datasets import generate
from ctrag.models import EdgeKind


def test_basinrag_upstream_is_commit_pinned() -> None:
    protocol = BasinRAGProtocol()
    assert protocol.upstream_repository == BASINRAG_REPOSITORY
    assert protocol.upstream_commit == BASINRAG_COMMIT
    assert len(BASINRAG_COMMIT) == 40
    assert BASINRAG_VERSION == "1.0.3"
    assert protocol.reranker.startswith("disabled")


def test_basinrag_reproduction_refuses_final_test_before_optional_dependency(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="restricted to train/dev"):
        run_basinrag_reproduction("test", tmp_path / "forbidden")


def test_no_causal_ablation_removes_only_explicit_causal_edges() -> None:
    dataset = generate("failure_recovery", seed=1234, traces=2)
    original_noncausal = sorted(
        (edge.source, edge.target, edge.kind.value)
        for source in dataset.topology.nodes
        for edge in dataset.topology.outgoing(source)
        if edge.kind is not EdgeKind.CAUSAL
    )
    ablated = _clone_without_causal_edges(dataset)
    assert set(ablated.nodes) == set(dataset.topology.nodes)
    assert not any(
        edge.kind is EdgeKind.CAUSAL
        for source in ablated.nodes
        for edge in ablated.outgoing(source)
    )
    remaining = sorted(
        (edge.source, edge.target, edge.kind.value)
        for source in ablated.nodes
        for edge in ablated.outgoing(source)
    )
    assert remaining == original_noncausal


def test_csv_handles_mixed_index_and_query_cost_rows(tmp_path: Path) -> None:
    path = tmp_path / "costs.csv"
    _csv(
        path,
        [
            {
                "dataset": "failure_recovery",
                "seed": 1,
                "arm": "basinrag_upstream_hybrid",
                "operation": "index",
                "elapsed_ms": 10.0,
            },
            {
                "dataset": "failure_recovery",
                "seed": 1,
                "query_id": "q-1",
                "arm": "basinrag_upstream_hybrid",
                "operation": "query",
                "elapsed_ms": 1.0,
            },
        ],
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["query_id"] == ""
    assert rows[1]["query_id"] == "q-1"
