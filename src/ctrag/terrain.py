from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .basins import AttractorDescriptor
from .models import CausalPath, CausalProvenance, Edge, EdgeKind
from .topology import BASIN_EDGE_KINDS, CausalTopology


EdgeIdentity = tuple[str, str, EdgeKind, CausalProvenance | None]


@dataclass(slots=True, frozen=True)
class TerrainConfig:
    """Configuration for the non-authoritative navigational terrain overlay."""

    reinforcement_step: float = 0.25
    decay_rate: float = 0.001
    minimum_influence: float = 0.0
    maximum_influence: float = 10.0

    def __post_init__(self) -> None:
        if self.reinforcement_step < 0 or not math.isfinite(self.reinforcement_step):
            raise ValueError("reinforcement_step must be finite and non-negative")
        if self.decay_rate < 0 or not math.isfinite(self.decay_rate):
            raise ValueError("decay_rate must be finite and non-negative")
        if self.minimum_influence < 0 or not math.isfinite(self.minimum_influence):
            raise ValueError("minimum_influence must be finite and non-negative")
        if self.maximum_influence <= 0 or not math.isfinite(self.maximum_influence):
            raise ValueError("maximum_influence must be finite and positive")
        if self.minimum_influence > self.maximum_influence:
            raise ValueError("minimum_influence cannot exceed maximum_influence")


@dataclass(slots=True, frozen=True)
class TerrainSnapshot:
    """Deterministic snapshot of navigational influence and basin membership."""

    transition_counts: dict[EdgeIdentity, int]
    influences: dict[EdgeIdentity, float]
    attractors: dict[str, AttractorDescriptor]
    basins: dict[str, frozenset[str]]


@dataclass(slots=True, frozen=True)
class BasinDrift:
    """Jaccard-distance drift between basin snapshots."""

    per_attractor: dict[str, float]
    mean: float


class DynamicTerrain:
    """Empirical terrain overlay learned from observed repeated trajectories.

    The overlay never mutates authoritative event history and never rewrites the
    graph's stored Edge weight/confidence. Reinforcement and erosion only change
    navigational influence maintained here.
    """

    def __init__(self, topology: CausalTopology, *, config: TerrainConfig | None = None) -> None:
        self.topology = topology
        self.config = config or TerrainConfig()
        self._transition_counts: dict[EdgeIdentity, int] = {}
        self._influence: dict[EdgeIdentity, float] = {}

    @staticmethod
    def edge_identity(edge: Edge) -> EdgeIdentity:
        return edge.identity()

    @property
    def transition_counts(self) -> dict[EdgeIdentity, int]:
        return dict(self._transition_counts)

    @property
    def influences(self) -> dict[EdgeIdentity, float]:
        return dict(self._influence)

    def transition_count(self, edge: Edge) -> int:
        return self._transition_counts.get(self.edge_identity(edge), 0)

    def influence(self, edge: Edge) -> float:
        """Return current navigational multiplier; unobserved edges default to 1."""
        return self._influence.get(self.edge_identity(edge), 1.0)

    def reinforce(self, edge: Edge, *, amount: float | None = None) -> float:
        """Record one observed transition and strengthen only its navigation weight."""
        if not self.topology.has_edge(edge):
            raise KeyError("cannot reinforce an edge that is not present in the topology")
        increment = self.config.reinforcement_step if amount is None else amount
        if increment < 0 or not math.isfinite(increment):
            raise ValueError("reinforcement amount must be finite and non-negative")
        identity = self.edge_identity(edge)
        self._transition_counts[identity] = self._transition_counts.get(identity, 0) + 1
        current = self._influence.get(identity, 1.0)
        updated = min(self.config.maximum_influence, current + increment)
        self._influence[identity] = updated
        return updated

    def _matching_edges(
        self,
        source: str,
        target: str,
        *,
        kind: EdgeKind,
        provenance: CausalProvenance | None = None,
    ) -> list[Edge]:
        edges = [
            edge
            for edge in self.topology.outgoing(source, {kind})
            if edge.target == target and (provenance is None or edge.provenance is provenance)
        ]
        return sorted(
            edges,
            key=lambda edge: (
                "" if edge.provenance is None else edge.provenance.value,
                edge.source,
                edge.target,
            ),
        )

    def observe_transition(
        self,
        source: str,
        target: str,
        *,
        kind: EdgeKind = EdgeKind.CAUSAL,
        provenance: CausalProvenance | None = None,
    ) -> float:
        edges = self._matching_edges(source, target, kind=kind, provenance=provenance)
        if not edges:
            raise KeyError(f"no {kind.value} edge exists for {source!r}->{target!r}")
        if len(edges) > 1 and provenance is None:
            raise ValueError("transition is ambiguous; specify causal provenance")
        return self.reinforce(edges[0])

    def observe_trajectory(
        self,
        node_ids: Iterable[str],
        *,
        kind: EdgeKind = EdgeKind.CAUSAL,
        provenance: CausalProvenance | None = None,
    ) -> None:
        nodes = list(node_ids)
        for source, target in zip(nodes, nodes[1:]):
            self.observe_transition(source, target, kind=kind, provenance=provenance)

    def decay(self, elapsed: float) -> None:
        """Erode navigation influence without deleting nodes, edges, or events."""
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError("elapsed must be finite and non-negative")
        factor = math.exp(-self.config.decay_rate * elapsed)
        for identity, current in list(self._influence.items()):
            self._influence[identity] = max(
                self.config.minimum_influence,
                current * factor,
            )

    def path_influence(self, path: CausalPath | None) -> float:
        """Geometric mean influence for a causal path, suitable as a score multiplier."""
        if path is None or not path.edges:
            return 1.0
        values = [max(1e-12, self.influence(edge)) for edge in path.edges]
        return math.exp(sum(math.log(value) for value in values) / len(values))

    def strongly_connected_components(
        self,
        *,
        kinds: Iterable[EdgeKind] = BASIN_EDGE_KINDS,
    ) -> tuple[tuple[str, ...], ...]:
        """Return deterministic Tarjan SCCs over selected topology edge kinds."""
        allowed = set(kinds)
        index = 0
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        components: list[tuple[str, ...]] = []

        def neighbors(node_id: str) -> list[str]:
            return sorted({
                edge.target
                for edge in self.topology.outgoing(node_id, allowed)
            })

        def visit(node_id: str) -> None:
            nonlocal index
            indices[node_id] = index
            lowlinks[node_id] = index
            index += 1
            stack.append(node_id)
            on_stack.add(node_id)

            for target in neighbors(node_id):
                if target not in indices:
                    visit(target)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node_id] = min(lowlinks[node_id], indices[target])

            if lowlinks[node_id] == indices[node_id]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node_id:
                        break
                components.append(tuple(sorted(component)))

        for node_id in sorted(self.topology.nodes):
            if node_id not in indices:
                visit(node_id)
        return tuple(sorted(components))

    def sinks(self, *, kinds: Iterable[EdgeKind] = BASIN_EDGE_KINDS) -> tuple[str, ...]:
        allowed = set(kinds)
        return tuple(
            node_id
            for node_id in sorted(self.topology.nodes)
            if not self.topology.outgoing(node_id, allowed)
        )

    def _incoming_observation_count(self, node_id: str) -> int:
        total = 0
        for edge in self.topology.incoming(node_id, BASIN_EDGE_KINDS):
            total += self.transition_count(edge)
        return total

    def _scc_observation_counts(self, members: set[str]) -> tuple[int, int]:
        internal = 0
        exits = 0
        for source in members:
            for edge in self.topology.outgoing(source, BASIN_EDGE_KINDS):
                count = self.transition_count(edge)
                if edge.target in members:
                    internal += count
                else:
                    exits += count
        return internal, exits

    def discover_attractors(self, *, register: bool = True) -> tuple[AttractorDescriptor, ...]:
        """Discover structural sinks and recurrent SCCs deterministically."""
        discovered: list[AttractorDescriptor] = []

        for node_id in self.sinks():
            visits = self._incoming_observation_count(node_id)
            confidence = min(1.0, 0.5 + 0.1 * visits)
            descriptor = AttractorDescriptor(
                node_id=node_id,
                confidence=confidence,
                origin="discovered:sink",
                metadata={"incoming_observations": visits},
            )
            discovered.append(descriptor)
            if register and self.topology.attractor(node_id) is None:
                self.topology.register_attractor(
                    node_id,
                    confidence=confidence,
                    origin=descriptor.origin,
                    metadata=descriptor.metadata,
                )

        for component in self.strongly_connected_components():
            members = set(component)
            recurrent = len(component) > 1
            if len(component) == 1:
                node_id = component[0]
                recurrent = any(
                    edge.target == node_id
                    for edge in self.topology.outgoing(node_id, BASIN_EDGE_KINDS)
                )
            if not recurrent:
                continue
            internal, exits = self._scc_observation_counts(members)
            ratio = internal / max(1, internal + exits)
            confidence = min(1.0, 0.5 + 0.5 * ratio)
            representative = component[0]
            descriptor = AttractorDescriptor(
                node_id=representative,
                confidence=confidence,
                origin="discovered:scc",
                metadata={
                    "members": list(component),
                    "internal_observations": internal,
                    "exit_observations": exits,
                },
            )
            discovered.append(descriptor)
            if register and self.topology.attractor(representative) is None:
                self.topology.register_attractor(
                    representative,
                    confidence=confidence,
                    origin=descriptor.origin,
                    metadata=descriptor.metadata,
                )

        unique = {(item.node_id, item.origin): item for item in discovered}
        return tuple(unique[key] for key in sorted(unique))

    def snapshot(self, *, max_hops: int = 8) -> TerrainSnapshot:
        attractors = {
            item.node_id: item
            for item in self.topology.attractor_descriptors()
        }
        basins = {
            node_id: frozenset(self.topology.basin(node_id, max_hops=max_hops))
            for node_id in sorted(attractors)
        }
        return TerrainSnapshot(
            transition_counts=dict(self._transition_counts),
            influences=dict(self._influence),
            attractors=attractors,
            basins=basins,
        )

    @staticmethod
    def basin_drift(before: TerrainSnapshot, after: TerrainSnapshot) -> BasinDrift:
        attractors = sorted(set(before.basins) | set(after.basins))
        per_attractor: dict[str, float] = {}
        for attractor_id in attractors:
            left = set(before.basins.get(attractor_id, frozenset()))
            right = set(after.basins.get(attractor_id, frozenset()))
            union = left | right
            drift = 0.0 if not union else 1.0 - (len(left & right) / len(union))
            per_attractor[attractor_id] = drift
        mean = sum(per_attractor.values()) / len(per_attractor) if per_attractor else 0.0
        return BasinDrift(per_attractor=per_attractor, mean=mean)
