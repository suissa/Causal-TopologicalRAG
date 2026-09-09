from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ctrag import (
    CausalProvenance,
    CausalTopology,
    CTRetriever,
    DynamicTerrain,
    Edge,
    EdgeEvidence,
    EdgeKind,
    MemoryNode,
    QueryMode,
)
from ctrag.storage import (
    EventSource,
    InMemoryVectorIndex,
    ListEventSource,
    MemoryStore,
    SQLiteCTStore,
    TerrainStore,
    TopologyStore,
    TopologyView,
    VectorIndex,
)
from ctrag.events import EventRecord


def build_topology() -> CausalTopology:
    topology = CausalTopology()
    now = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)
    topology.add_node(MemoryNode(
        id="cause",
        text="payment authorized by provider",
        timestamp=now,
        metadata={"execution_id": "exec-1", "status": "ok"},
        embedding=(1.0, 0.0, 0.0),
    ))
    topology.add_node(MemoryNode(
        id="error",
        text="inventory reservation failed",
        timestamp=now + timedelta(seconds=1),
        metadata={"execution_id": "exec-1", "status": "error"},
        embedding=(0.0, 1.0, 0.0),
    ))
    topology.add_node(MemoryNode(
        id="healed",
        text="inventory recovered after refresh",
        timestamp=now + timedelta(seconds=2),
        metadata={"execution_id": "exec-1", "status": "healed"},
        embedding=(0.0, 0.0, 1.0),
    ))
    topology.add_edge(Edge(
        source="cause",
        target="error",
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.EXECUTION,
        confidence=0.95,
        weight=0.8,
        evidence=(EdgeEvidence(
            id="ev-1",
            source="event.causation_id",
            metadata={"event_id": "error"},
        ),),
        provenance_metadata={"source_system": "fixture"},
    ))
    topology.add_edge(Edge(
        source="error",
        target="healed",
        kind=EdgeKind.CAUSAL,
        provenance=CausalProvenance.WORKFLOW,
        confidence=0.9,
        evidence=(EdgeEvidence(id="ev-2", source="workflow"),),
    ))
    topology.add_edge(Edge(
        source="cause",
        target="error",
        kind=EdgeKind.TEMPORAL,
    ))
    topology.register_attractor(
        "healed",
        confidence=0.88,
        origin="manual",
        metadata={"reason": "successful recovery"},
    )
    return topology


def topology_payload(topology: CausalTopology) -> tuple[list[dict], list[dict], list[dict]]:
    nodes = [topology.nodes[node_id].to_dict() for node_id in sorted(topology.nodes)]
    edges = []
    for node_id in sorted(topology.nodes):
        edges.extend(edge.to_dict() for edge in topology.outgoing(node_id))
    edges.sort(key=lambda item: (
        item["source"], item["target"], item["kind"], item["provenance"] or ""
    ))
    attractors = [item.to_dict() for item in topology.attractor_descriptors()]
    return nodes, edges, attractors


def test_reference_topology_satisfies_topology_view_protocol() -> None:
    topology = build_topology()
    assert isinstance(topology, TopologyView)


def test_sqlite_store_conforms_to_declared_store_protocols(tmp_path) -> None:
    store = SQLiteCTStore(tmp_path / "ctrag.sqlite")
    assert isinstance(store, MemoryStore)
    assert isinstance(store, TopologyStore)
    assert isinstance(store, TerrainStore)

    vectors = InMemoryVectorIndex()
    assert isinstance(vectors, VectorIndex)

    events = ListEventSource([
        EventRecord(
            event_id="e1",
            event_type="Test.Event",
            timestamp=datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc),
        )
    ])
    assert isinstance(events, EventSource)
    assert [event.event_id for event in events.read_events()] == ["e1"]


def test_memory_store_round_trip(tmp_path) -> None:
    store = SQLiteCTStore(tmp_path / "memory.sqlite")
    node = build_topology().nodes["cause"]
    store.put_memory(node)

    loaded = store.get_memory("cause")
    assert loaded is not None
    assert loaded.to_dict() == node.to_dict()
    assert [item.to_dict() for item in store.list_memories()] == [node.to_dict()]


def test_sqlite_topology_round_trip_preserves_all_semantic_inputs(tmp_path) -> None:
    original = build_topology()
    store = SQLiteCTStore(tmp_path / "projection.sqlite")
    store.save_topology(original)

    loaded = store.load_topology()

    assert topology_payload(loaded) == topology_payload(original)
    causal = loaded.outgoing("cause", {EdgeKind.CAUSAL})[0]
    assert causal.provenance is CausalProvenance.EXECUTION
    assert causal.confidence == 0.95
    assert causal.weight == 0.8
    assert causal.evidence[0].id == "ev-1"
    assert causal.provenance_metadata == {"source_system": "fixture"}
    assert loaded.attractor("healed").to_dict() == original.attractor("healed").to_dict()


def test_reload_preserves_ctrag_ranking_inputs_and_results(tmp_path) -> None:
    original = build_topology()
    store = SQLiteCTStore(tmp_path / "rank.sqlite")

    before = CTRetriever(original).search(
        "why did inventory reservation fail?",
        mode=QueryMode.WHY,
        anchor_ids=["error"],
        k=2,
        exhaustive=True,
    )
    store.save_topology(original)
    loaded = store.load_topology()
    after = CTRetriever(loaded).search(
        "why did inventory reservation fail?",
        mode=QueryMode.WHY,
        anchor_ids=["error"],
        k=2,
        exhaustive=True,
    )

    assert [hit.node.id for hit in after] == [hit.node.id for hit in before]
    assert [hit.score for hit in after] == [hit.score for hit in before]
    assert [hit.components for hit in after] == [hit.components for hit in before]
    assert [hit.causal_hops for hit in after] == [hit.causal_hops for hit in before]


def test_terrain_overlay_round_trip_is_exact(tmp_path) -> None:
    topology = build_topology()
    terrain = DynamicTerrain(topology)
    terrain.observe_transition(
        "cause",
        "error",
        provenance=CausalProvenance.EXECUTION,
    )
    terrain.observe_transition(
        "cause",
        "error",
        provenance=CausalProvenance.EXECUTION,
    )
    terrain.observe_transition(
        "error",
        "healed",
        provenance=CausalProvenance.WORKFLOW,
    )
    terrain.decay(10.0)

    store = SQLiteCTStore(tmp_path / "terrain.sqlite")
    store.save_topology(topology)
    store.save_terrain(terrain)
    loaded_topology = store.load_topology()
    loaded_terrain = store.load_terrain(loaded_topology)

    assert loaded_terrain.config == terrain.config
    assert loaded_terrain.transition_counts == terrain.transition_counts
    assert loaded_terrain.influences == terrain.influences


def test_in_memory_vector_index_is_deterministic() -> None:
    index = InMemoryVectorIndex()
    index.upsert_vector("b", (1.0, 0.0))
    index.upsert_vector("a", (1.0, 0.0))
    index.upsert_vector("c", (0.0, 1.0))

    assert index.search_vector((1.0, 0.0), k=3) == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 0.0),
    ]
