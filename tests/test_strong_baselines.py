from __future__ import annotations

import string
from pathlib import Path

import pytest

from ctrag.adapters import RankBM25Retriever
from ctrag.benchmarks.strong_baselines import (
    CachedEmbedder,
    STRONG_DENSE_MODELS,
    run_strong_baselines,
)


def test_strong_dense_models_are_revision_pinned_and_dimension_distinct() -> None:
    assert len(STRONG_DENSE_MODELS) >= 2
    dimensions = set()
    for spec in STRONG_DENSE_MODELS:
        assert spec.model_name.startswith("sentence-transformers/")
        assert len(spec.revision) == 40
        assert all(char in string.hexdigits for char in spec.revision)
        assert spec.dimensions > 0
        assert spec.normalize_embeddings is True
        dimensions.add(spec.dimensions)
    assert len(dimensions) >= 2


def test_strong_baseline_harness_refuses_final_test(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="restricted to train/dev"):
        run_strong_baselines("test", tmp_path / "forbidden", ks=(1,))


def test_rank_bm25_descriptor_freezes_parameters_without_optional_dependency() -> None:
    descriptor = RankBM25Retriever(k1=1.5, b=0.75, epsilon=0.25).descriptor()
    assert descriptor["name"] == "BM25Okapi"
    assert descriptor["implementation"] == "rank-bm25"
    assert descriptor["k1"] == 1.5
    assert descriptor["b"] == 0.75
    assert descriptor["epsilon"] == 0.25


def test_cached_embedder_invalidation_forces_fresh_query_embedding() -> None:
    class CountingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> tuple[float, ...]:
            self.calls += 1
            return (float(self.calls), float(len(text)))

    delegate = CountingEmbedder()
    cached = CachedEmbedder(delegate)

    first = cached.embed("query")
    assert cached.embed("query") == first
    assert delegate.calls == 1

    cached.invalidate("query")
    second = cached.embed("query")
    assert delegate.calls == 2
    assert second != first
