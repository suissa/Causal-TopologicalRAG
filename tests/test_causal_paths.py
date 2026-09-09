from datetime import datetime, timezone

import pytest

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    Edge,
    EdgeEvidence,
    EdgeKind,
    MemoryNode,
    QueryMode,
    RetrievalWeights,
)


def _add_nodes(topology: CausalTopology, *node_ids: str) -> None:
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    for node_id in node_ids:
        topology.add_node(MemoryNode(id=node_id, text="same evidence text", timestamp=now))


def test_multi_path_confidence_aggregates_and_best_path_is_reconstructable() -> None:
    topology = CausalTopology()
    _add_nodes(topology, "anchor", "b", "c", "target")
    topology.add_edge(Edge(
        source="anchor", target="b", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION, confidence=0.9,
    ))
    topology.add_edge(Edge(
        source="b", target="target", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    topology.add_edge(Edge(
        source="anchor", target="c", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.INFERRED,
    ))
    topology.add_edge(Edge(
        source="c", target="target", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    topology.add_edge(Edge(
        source="c", target="anchor", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))

    path = topology.causal_path_evidence("anchor", "target", direction="out", max_hops=6)

    assert path is not None
    assert path.nodes == ("anchor", "b", "target")
    assert path.best_confidence == pytest.approx(0.9)
    assert path.aggregate_confidence == pytest.approx(0.96)
    assert path.hops == 2


def test_observed_execution_edge_ranks_above_equivalent_inferred_edge() -> None:
    topology = CausalTopology()
    _add_nodes(topology, "anchor", "observed", "inferred")
    topology.add_edge(Edge(
        source="anchor", target="observed", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    topology.add_edge(Edge(
        source="anchor", target="inferred", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.INFERRED,
    ))

    retriever = CTRetriever(topology)
    hits = retriever.search(
        "same evidence text",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["anchor"],
        weights=RetrievalWeights(0, 0, 1, 0, 0, 0),
        exhaustive=True,
        k=2,
    )

    assert [hit.node.id for hit in hits] == ["observed", "inferred"]
    assert hits[0].components["causal"] == pytest.approx(1.0)
    assert hits[1].components["causal"] == pytest.approx(0.6)


def test_temporal_and_behavioral_shortcuts_do_not_contribute_to_causal_path() -> None:
    topology = CausalTopology()
    _add_nodes(topology, "anchor", "mid", "target")
    topology.add_edge(Edge(source="anchor", target="target", kind=EdgeKind.TEMPORAL))
    topology.add_edge(Edge(source="anchor", target="target", kind=EdgeKind.BEHAVIORAL))
    topology.add_edge(Edge(
        source="anchor", target="mid", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    topology.add_edge(Edge(
        source="mid", target="target", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))

    path = topology.causal_path_evidence("anchor", "target", direction="out")

    assert path is not None
    assert path.nodes == ("anchor", "mid", "target")
    assert path.hops == 2
    assert all(edge.kind is EdgeKind.CAUSAL for edge in path.edges)


def test_retrieval_hit_exposes_selected_causal_path_and_evidence() -> None:
    topology = CausalTopology()
    _add_nodes(topology, "anchor", "target")
    topology.add_edge(Edge(
        source="anchor",
        target="target",
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EVENT,
        evidence=(EdgeEvidence(id="event-store:42", source="event-store"),),
    ))

    hit = CTRetriever(topology).search(
        "same evidence text",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["anchor"],
        weights=RetrievalWeights(0, 0, 1, 0, 0, 0),
        k=1,
    )[0]

    assert hit.causal_path is not None
    assert hit.causal_path.nodes == ("anchor", "target")
    assert hit.causal_path.evidence[0].id == "event-store:42"
    assert hit.causal_path.provenances == (CausalProvenance.EVENT,)


def test_ancestor_and_descendant_budgets_are_direction_specific() -> None:
    ancestors = CausalTopology()
    _add_nodes(ancestors, "root", "mid", "anchor")
    ancestors.add_edge(Edge(
        source="root", target="mid", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    ancestors.add_edge(Edge(
        source="mid", target="anchor", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    why_hits = CTRetriever(ancestors).search(
        "same evidence text",
        mode=QueryMode.WHY,
        anchor_ids=["anchor"],
        ancestor_hops=1,
        weights=RetrievalWeights(0, 0, 1, 0, 0, 0),
        exhaustive=True,
        k=2,
    )
    why_by_id = {hit.node.id: hit for hit in why_hits}
    assert why_by_id["mid"].components["causal"] > 0
    assert why_by_id["root"].components["causal"] == 0

    descendants = CausalTopology()
    _add_nodes(descendants, "anchor", "mid", "leaf")
    descendants.add_edge(Edge(
        source="anchor", target="mid", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    descendants.add_edge(Edge(
        source="mid", target="leaf", kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
    ))
    next_hits = CTRetriever(descendants).search(
        "same evidence text",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["anchor"],
        descendant_hops=1,
        weights=RetrievalWeights(0, 0, 1, 0, 0, 0),
        exhaustive=True,
        k=2,
    )
    next_by_id = {hit.node.id: hit for hit in next_hits}
    assert next_by_id["mid"].components["causal"] > 0
    assert next_by_id["leaf"].components["causal"] == 0
