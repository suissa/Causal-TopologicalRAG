from datetime import datetime, timezone

import pytest

from ctrag import (
    BM25Retriever,
    CausalTopology,
    CTRetriever,
    EmbeddingProvider,
    IdfOverlapRetriever,
    LexicalRetriever,
    MemoryNode,
    reciprocal_rank_fusion,
)
from ctrag.embedding import HashingEmbedder


def _topology() -> CausalTopology:
    topology = CausalTopology()
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    topology.add_node(MemoryNode(id="a", text="payment authorization approved", timestamp=now))
    topology.add_node(MemoryNode(id="b", text="inventory reservation failed", timestamp=now))
    topology.add_node(MemoryNode(id="c", text="payment authorization retry", timestamp=now))
    return topology


def test_protocols_accept_builtin_adapters() -> None:
    assert isinstance(HashingEmbedder(), EmbeddingProvider)
    assert isinstance(IdfOverlapRetriever(), LexicalRetriever)
    assert isinstance(BM25Retriever(), LexicalRetriever)


def test_bm25_prefers_repeated_relevant_terms() -> None:
    retriever = BM25Retriever()
    scores = retriever.score(
        "payment authorization",
        {
            "a": "payment authorization approved",
            "b": "inventory reservation failed",
            "c": "payment payment authorization retry",
        },
    )

    assert scores["c"] > scores["a"] > scores["b"]
    assert max(scores.values()) == pytest.approx(1.0)


def test_rrf_is_deterministic_and_uses_first_duplicate_rank_only() -> None:
    rankings = [["b", "a", "a", "c"], ["a", "b", "c"]]

    first = reciprocal_rank_fusion(rankings, rank_constant=10)
    second = reciprocal_rank_fusion(rankings, rank_constant=10)

    assert first == second
    assert [node_id for node_id, _ in first] == ["b", "a", "c"]


def test_dense_lexical_and_hybrid_can_be_ranked_independently() -> None:
    retriever = CTRetriever(_topology(), lexical_retriever=BM25Retriever())

    dense = retriever.rank_dense("payment authorization", k=3)
    lexical = retriever.rank_lexical("payment authorization", k=3)
    hybrid = retriever.rank_hybrid_rrf("payment authorization", k=3, rank_constant=10)

    assert len(dense) == len(lexical) == len(hybrid) == 3
    assert {node_id for node_id, _ in dense} == {"a", "b", "c"}
    assert {node_id for node_id, _ in lexical} == {"a", "b", "c"}
    assert {node_id for node_id, _ in hybrid} == {"a", "b", "c"}


def test_default_lexical_adapter_preserves_original_overlap_behavior() -> None:
    retriever = CTRetriever(_topology())

    scores = dict(retriever.rank_lexical("inventory failed", k=3))

    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_invalid_bm25_parameters_fail_fast() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(k1=0)
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)
