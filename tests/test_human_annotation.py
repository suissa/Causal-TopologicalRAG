from ctrag.benchmarks.human_annotation import agreement, cohen_kappa, make_blinded_sample


def test_blinded_sample_hides_system_identity_and_is_deterministic() -> None:
    rows = [
        {"dataset": "d", "query_id": "q", "arm": "secret-system-a", "retrieved_ids": ["n1", "n2"]},
        {"dataset": "d", "query_id": "q", "arm": "secret-system-b", "retrieved_ids": ["n2", "n3"]},
    ]
    first = make_blinded_sample(rows, seed=7, max_items=10)
    second = make_blinded_sample(rows, seed=7, max_items=10)
    assert first == second
    assert first
    assert all("arm" not in item for item in first)


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == 1.0


def test_agreement_is_computed_before_adjudication() -> None:
    rows = []
    for item_id, a, b in (("a", 2, 2), ("b", 0, 1)):
        for annotator, value in (("ann-1", a), ("ann-2", b)):
            rows.append({
                "item_id": item_id,
                "annotator_id": annotator,
                "relevance": value,
                "causal_support": value,
                "recovery_usefulness": value,
                "unsupported_claim": 0,
            })
    result = agreement(rows)
    assert result["paired_items"] == 2
    assert set(result["agreement_before_adjudication"]) == {
        "relevance", "causal_support", "recovery_usefulness", "unsupported_claim"
    }
