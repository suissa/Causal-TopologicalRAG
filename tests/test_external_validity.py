from pathlib import Path

from ctrag.benchmarks.external_validity import load_sources, run, source_to_dataset
from ctrag.models import EdgeKind

FIXTURE = Path("research/external/github-actions-v1.json")


def test_external_fixture_has_two_independent_public_sources() -> None:
    sources = load_sources(FIXTURE)
    independent = [source for source in sources if source.independent_public]
    assert len(independent) >= 2
    assert {source.repository for source in independent} >= {"Basinfy/BasinRAG", "psf/requests"}


def test_step_order_never_becomes_causal_edge() -> None:
    for source in load_sources(FIXTURE):
        dataset = source_to_dataset(source)
        edges = [edge for node_id in dataset.topology.nodes for edge in dataset.topology.outgoing(node_id)]
        assert edges
        assert all(edge.kind is EdgeKind.TEMPORAL for edge in edges)
        assert not any(edge.kind is EdgeKind.CAUSAL for edge in edges)


def test_external_queries_use_observed_targets_not_causal_labels() -> None:
    for source in load_sources(FIXTURE):
        dataset = source_to_dataset(source)
        assert dataset.queries
        for query in dataset.queries:
            assert query.relevance
            assert query.causal_nodes == []
            assert query.causal_paths == []
            assert query.causal_distances == {}


def test_external_benchmark_is_reproducible_and_does_not_unblind_holdout(tmp_path: Path) -> None:
    first = run(tmp_path / "a", FIXTURE, k=3)
    second = run(tmp_path / "b", FIXTURE, k=3)
    assert first == second
    assert first["final_holdout_executed"] is False
    assert len(first["datasets"]) == 4
