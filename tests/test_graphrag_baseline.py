from __future__ import annotations

from pathlib import Path

import pytest

from ctrag.adapters import BM25Retriever
from ctrag.benchmarks.datasets import generate
from ctrag.benchmarks.graphrag_baseline import (
    RelationGraphConfig,
    SemanticRelationGraph,
    run_graphrag_comparison,
)
from ctrag.directional_retriever import CTRetriever
from ctrag.embedding import HashingEmbedder
from ctrag.models import CausalProvenance, Edge, EdgeKind


def _relation_graph(dataset):
    retriever = CTRetriever(
        dataset.topology,
        embedder=HashingEmbedder(128),
        lexical_retriever=BM25Retriever(),
    )
    return SemanticRelationGraph(retriever, neighbors=3)


def test_semantic_relation_graph_is_invariant_to_extra_causal_edges() -> None:
    left = generate("failure_recovery", seed=7, traces=2)
    right = generate("failure_recovery", seed=7, traces=2)
    ids = sorted(right.topology.nodes)
    right.topology.add_edge(
        Edge(ids[-1], ids[0], EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)
    )

    assert _relation_graph(left).adjacency == _relation_graph(right).adjacency


def test_graphrag_comparison_refuses_final_test(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="restricted to train/dev"):
        run_graphrag_comparison("test", tmp_path / "forbidden")


def test_relation_graph_config_rejects_zero_budget() -> None:
    with pytest.raises(ValueError):
        RelationGraphConfig(neighbors=0)
    with pytest.raises(ValueError):
        RelationGraphConfig(max_hops=0)
