from __future__ import annotations

from pathlib import Path

import pytest

from ctrag.benchmarks.anchor_discovery import ANCHOR_K, _ambiguous_queries, run_anchor_discovery
from ctrag.benchmarks.datasets import generate


def test_no_oracle_anchor_harness_refuses_final_test(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="restricted to train/dev"):
        run_anchor_discovery("test", tmp_path / "forbidden")


def test_raw_queries_do_not_expose_opaque_generated_node_ids() -> None:
    dataset = generate("failure_recovery", seed=7, traces=4)
    node_ids = set(dataset.topology.nodes)
    assert all(not any(node_id in query.text for node_id in node_ids) for query in dataset.queries)


def test_ambiguous_anchor_controls_have_multiple_valid_anchors() -> None:
    dataset = generate("failure_recovery", seed=7, traces=4)
    ambiguous = _ambiguous_queries(dataset.topology)
    assert ambiguous
    assert ANCHOR_K == 3
    assert all(len(row["gold_anchor_ids"]) >= 2 for row in ambiguous)
    assert any(row["text"] == "request accepted" for row in ambiguous)
