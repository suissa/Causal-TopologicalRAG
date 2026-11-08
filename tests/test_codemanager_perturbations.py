from ctrag.benchmarks.codemanager_perturbations import evaluate, fixture, perturbations


def test_perturbation_suite_is_deterministic_and_fixed_denominator():
    scenarios = perturbations(fixture())
    first = {name: evaluate(items) for name, items in scenarios.items()}
    second = {name: evaluate(items) for name, items in scenarios.items()}
    assert first == second
    assert len(first) == 9
    assert all(0.0 <= row["mrr"] <= 1.0 for row in first.values())


def test_reverse_order_does_not_change_rank():
    scenarios = perturbations(fixture())
    assert evaluate(scenarios["identity"]) == evaluate(scenarios["reverse_input"])


def test_duplicate_same_source_does_not_inflate_score():
    scenarios = perturbations(fixture())
    original = evaluate(scenarios["identity"])["scores"]
    duplicated = evaluate(scenarios["same_source_duplicate"])["scores"]
    assert original == duplicated


def test_cross_evidence_group_cannot_support_candidate():
    scenarios = perturbations(fixture())
    original = evaluate(scenarios["identity"])["scores"]
    leak = evaluate(scenarios["cross_group_leak_attempt"])["scores"]
    assert original == leak
