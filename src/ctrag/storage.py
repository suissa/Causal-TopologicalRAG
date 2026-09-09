from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from .basins import AttractorDescriptor, BasinAffinity
from .embedding import cosine_similarity
from .events import EventRecord
from .models import CausalPath, CausalProvenance, Edge, EdgeKind, MemoryNode
from .terrain import DynamicTerrain, EdgeIdentity, TerrainConfig
from .topology import CausalTopology


@runtime_checkable
class TopologyView(Protocol):
    """Read/navigation contract consumed by the CT-RAG retriever.

    A database-backed implementation may satisfy this protocol directly. The
    built-in `CausalTopology` is the in-memory reference implementation.
    """

    nodes: Mapping[str, MemoryNode]

    def outgoing(self, node_id: str, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]: ...
    def incoming(self, node_id: str, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]: ...
    def distances(
        self,
        node_id: str,
        *,
        direction: str = "both",
        kinds: Iterable[EdgeKind] | None = None,
        max_hops: int = 4,
    ) -> dict[str, int]: ...
    def neighborhood(
        self,
        node_id: str,
        *,
        direction: str = "both",
        kinds: Iterable[EdgeKind] | None = None,
        max_hops: int = 4,
        include_anchor: bool = False,
    ) -> set[str]: ...
    def causal_path_evidence(
        self,
        anchor_id: str,
        candidate_id: str,
        *,
        direction: str,
        max_hops: int = 4,
    ) -> CausalPath | None: ...
    def basin_memberships(self, node_id: str, max_hops: int = 8) -> set[str]: ...
    def basin(self, attractor_id: str, max_hops: int = 8) -> set[str]: ...
    def shared_basin_affinity(
        self,
        left_id: str,
        right_id: str,
        *,
        max_hops: int = 8,
    ) -> BasinAffinity: ...


@runtime_checkable
class MemoryStore(Protocol):
    def put_memory(self, node: MemoryNode) -> None: ...
    def get_memory(self, node_id: str) -> MemoryNode | None: ...
    def list_memories(self) -> list[MemoryNode]: ...


@runtime_checkable
class VectorIndex(Protocol):
    def upsert_vector(self, node_id: str, vector: Iterable[float]) -> None: ...
    def delete_vector(self, node_id: str) -> None: ...
    def search_vector(self, vector: Iterable[float], *, k: int = 10) -> list[tuple[str, float]]: ...


@runtime_checkable
class TopologyStore(Protocol):
    def save_topology(self, topology: CausalTopology) -> None: ...
    def load_topology(self) -> CausalTopology: ...


@runtime_checkable
class EventSource(Protocol):
    """Authoritative event source contract; separate from retrieval projections."""

    def read_events(self) -> Iterable[EventRecord]: ...


@runtime_checkable
class TerrainStore(Protocol):
    def save_terrain(self, terrain: DynamicTerrain) -> None: ...
    def load_terrain(self, topology: CausalTopology) -> DynamicTerrain: ...


class InMemoryVectorIndex:
    """Deterministic dependency-free reference VectorIndex."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[float, ...]] = {}

    def upsert_vector(self, node_id: str, vector: Iterable[float]) -> None:
        values = tuple(float(value) for value in vector)
        if not node_id.strip():
            raise ValueError("node_id must be non-empty")
        self._vectors[node_id] = values

    def delete_vector(self, node_id: str) -> None:
        self._vectors.pop(node_id, None)

    def search_vector(self, vector: Iterable[float], *, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = tuple(float(value) for value in vector)
        ranked = [
            (node_id, cosine_similarity(query, candidate))
            for node_id, candidate in self._vectors.items()
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]


class ListEventSource:
    """Small in-memory EventSource useful for tests and adapters."""

    def __init__(self, events: Iterable[EventRecord]) -> None:
        self._events = tuple(events)

    def read_events(self) -> Iterable[EventRecord]:
        return iter(self._events)


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _identity_to_dict(identity: EdgeIdentity) -> dict[str, str | None]:
    source, target, kind, provenance = identity
    return {
        "source": source,
        "target": target,
        "kind": kind.value,
        "provenance": None if provenance is None else provenance.value,
    }


def _identity_from_dict(raw: dict[str, object]) -> EdgeIdentity:
    provenance = raw.get("provenance")
    return (
        str(raw["source"]),
        str(raw["target"]),
        EdgeKind(str(raw["kind"])),
        None if provenance is None else CausalProvenance(str(provenance)),
    )


class SQLiteCTStore:
    """Persistent local projection store implemented only with stdlib SQLite.

    This adapter stores retrieval projections, not the authoritative event log.
    It implements MemoryStore, TopologyStore and TerrainStore while preserving
    serialized ranking inputs exactly.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    seq INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attractors (
                    node_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS terrain_state (
                    name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                """
            )

    # MemoryStore ---------------------------------------------------------
    def put_memory(self, node: MemoryNode) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO memories(id, payload) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (node.id, _json_dumps(node.to_dict())),
            )

    def get_memory(self, node_id: str) -> MemoryNode | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM memories WHERE id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        return MemoryNode.from_dict(json.loads(row[0]))

    def list_memories(self) -> list[MemoryNode]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM memories ORDER BY id"
            ).fetchall()
        return [MemoryNode.from_dict(json.loads(row[0])) for row in rows]

    # TopologyStore -------------------------------------------------------
    def save_topology(self, topology: CausalTopology) -> None:
        edges: list[Edge] = []
        for node_id in sorted(topology.nodes):
            edges.extend(topology.outgoing(node_id))
        edges.sort(
            key=lambda edge: (
                edge.source,
                edge.target,
                edge.kind.value,
                "" if edge.provenance is None else edge.provenance.value,
            )
        )

        with self._connect() as connection:
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM attractors")
            connection.execute("DELETE FROM memories")
            connection.executemany(
                "INSERT INTO memories(id, payload) VALUES (?, ?)",
                [
                    (node_id, _json_dumps(topology.nodes[node_id].to_dict()))
                    for node_id in sorted(topology.nodes)
                ],
            )
            connection.executemany(
                "INSERT INTO edges(seq, payload) VALUES (?, ?)",
                [
                    (index, _json_dumps(edge.to_dict()))
                    for index, edge in enumerate(edges)
                ],
            )
            connection.executemany(
                "INSERT INTO attractors(node_id, payload) VALUES (?, ?)",
                [
                    (descriptor.node_id, _json_dumps(descriptor.to_dict()))
                    for descriptor in topology.attractor_descriptors()
                ],
            )

    def load_topology(self) -> CausalTopology:
        topology = CausalTopology()
        with self._connect() as connection:
            node_rows = connection.execute(
                "SELECT payload FROM memories ORDER BY id"
            ).fetchall()
            edge_rows = connection.execute(
                "SELECT payload FROM edges ORDER BY seq"
            ).fetchall()
            attractor_rows = connection.execute(
                "SELECT payload FROM attractors ORDER BY node_id"
            ).fetchall()

        for row in node_rows:
            topology.add_node(MemoryNode.from_dict(json.loads(row[0])))
        for row in edge_rows:
            topology.add_edge(Edge.from_dict(json.loads(row[0])))
        for row in attractor_rows:
            descriptor = AttractorDescriptor.from_dict(json.loads(row[0]))
            topology.register_attractor(
                descriptor.node_id,
                confidence=descriptor.confidence,
                origin=descriptor.origin,
                metadata=descriptor.metadata,
            )
        return topology

    # TerrainStore --------------------------------------------------------
    def save_terrain(self, terrain: DynamicTerrain) -> None:
        payload = {
            "config": {
                "reinforcement_step": terrain.config.reinforcement_step,
                "decay_rate": terrain.config.decay_rate,
                "minimum_influence": terrain.config.minimum_influence,
                "maximum_influence": terrain.config.maximum_influence,
            },
            "transitions": [
                {
                    "edge": _identity_to_dict(identity),
                    "count": count,
                    "influence": terrain.influences.get(identity, 1.0),
                }
                for identity, count in sorted(
                    terrain.transition_counts.items(),
                    key=lambda item: _json_dumps(_identity_to_dict(item[0])),
                )
            ],
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO terrain_state(name, payload) VALUES ('overlay', ?) "
                "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
                (_json_dumps(payload),),
            )

    def load_terrain(self, topology: CausalTopology) -> DynamicTerrain:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM terrain_state WHERE name='overlay'"
            ).fetchone()
        if row is None:
            return DynamicTerrain(topology)

        payload = json.loads(row[0])
        config = TerrainConfig(**payload["config"])
        terrain = DynamicTerrain(topology, config=config)
        # Internal restoration is intentionally isolated in this adapter; the
        # values remain an overlay and never rewrite stored topology edges.
        transition_counts: dict[EdgeIdentity, int] = {}
        influences: dict[EdgeIdentity, float] = {}
        for item in payload.get("transitions", []):
            identity = _identity_from_dict(item["edge"])
            transition_counts[identity] = int(item["count"])
            influences[identity] = float(item["influence"])
        terrain._transition_counts = transition_counts
        terrain._influence = influences
        return terrain


__all__ = [
    "EventSource",
    "InMemoryVectorIndex",
    "ListEventSource",
    "MemoryStore",
    "SQLiteCTStore",
    "TerrainStore",
    "TopologyStore",
    "TopologyView",
    "VectorIndex",
]
