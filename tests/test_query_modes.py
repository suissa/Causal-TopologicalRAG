from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    Edge,
    EdgeKind,
    MemoryNode,
    QueryMode,
)


def build_trace() -> tuple[CausalTopology, CTRetriever]:
    topology = CausalTopology()
    start = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)
    nodes = [
        ("decision", "checkout selected payment path", "ok"),
        ("payment", "payment authorized", "ok"),
        ("error", "inventory reservation failed stock changed", "error"),
        ("heal", "inventory healed by refreshing stock", "healed"),
        ("recovered", "checkout completed after recovery", "completed"),
        ("alternate", "checkout selected alternate reservation path", "ok"),
        ("alternate-success", "alternate checkout completed", "completed"),
    ]
    for index, (node_id, text, status) in enumerate(nodes):
        topology.add_node(MemoryNode(
            id=node_id,
            text=text,
            timestamp=start + timedelta(seconds=index),
            metadata={
                "status": status,
                "event_type": text.replace(" ", "."),
                "execution_id": "exec-main" if not node_id.startswith("alternate") else "exec-alt",
                "intent_id": "checkout",
            },
        ))

    for source, target in (
        ("decision", "payment"),
        ("payment", "error"),
        ("error", "heal"),
        ("heal", "recovered"),
        ("decision", "alternate"),
        ("alternate", "alternate-success"),
    ):
        topology.add_edge(Edge(
            source=source,
            target=target,
            kind=EdgeKind.CAUSAL,
            provenance=CausalProvenance.EXECUTION,
        ))

    topology.register_attractor("recovered", confidence=1.0, origin="manual")
    topology.register_attractor("alternate-success", confidence=1.0, origin="manual")
    return topology, CTRetriever(topology)


def stage_nodes(result, name: str) -> set[str]:
    stage = next(stage for stage in result.stages if stage.name == name)
    return set(stage.node_ids)


def test_why_traverses_only_causal_ancestors() -> None:
    _, retriever = build_trace()
    result = retriever.search_staged(
        "why did inventory reservation fail?",
        mode=QueryMode.WHY,
        anchor_ids=["error"],
        k=4,
    )

    assert result.direction == "in"
    causal = stage_nodes(result, "causal_traversal")
    assert {"payment", "decision"}.issubset(causal)
    assert "heal" not in causal
    assert "recovered" not in causal
    assert result.selected_anchors == ("error",)
    assert any(hit.anchor_id == "error" and hit.causal_hops is not None for hit in result.hits)
    assert all("causal" in hit.components and "topological" in hit.components for hit in result.hits)


def test_what_next_traverses_only_causal_descendants() -> None:
    _, retriever = build_trace()
    result = retriever.search_staged(
        "what happens after the inventory failure?",
        mode=QueryMode.WHAT_NEXT,
        anchor_ids=["error"],
        k=4,
    )

    assert result.direction == "out"
    causal = stage_nodes(result, "causal_traversal")
    assert {"heal", "recovered"}.issubset(causal)
    assert "payment" not in causal
    assert "decision" not in causal


def test_recovery_prioritizes_successful_observed_future_path() -> None:
    _, retriever = build_trace()
    result = retriever.recovery("error", k=4)

    assert result.mode is QueryMode.RECOVERY
    assert result.direction == "out"
    assert result.hits[0].node.id == "heal"
    assert result.hits[0].components["mode_prior"] > 0.0
    assert {"heal", "recovered"}.intersection(stage_nodes(result, "mode_rerank"))


def test_counterfactual_surfaces_historical_divergence_with_disclaimer() -> None:
    _, retriever = build_trace()
    result = retriever.counterfactual("error", k=5)

    assert result.mode is QueryMode.COUNTERFACTUAL
    assert result.direction == "in"
    assert result.observational_note is not None
    assert "observational" in result.observational_note.lower()
    assert "does not identify intervention effects" in result.observational_note.lower()

    decision = next(hit for hit in result.hits if hit.node.id == "decision")
    assert decision.components["mode_prior"] > 0.0
    assert "decision" in stage_nodes(result, "mode_rerank")


def test_same_trace_changes_context_by_query_mode() -> None:
    _, retriever = build_trace()
    why = retriever.why("error", k=4)
    what_next = retriever.what_next("error", k=4)

    why_causal = stage_nodes(why, "causal_traversal")
    next_causal = stage_nodes(what_next, "causal_traversal")
    assert why_causal != next_causal
    assert "payment" in why_causal and "payment" not in next_causal
    assert "heal" in next_causal and "heal" not in why_causal
