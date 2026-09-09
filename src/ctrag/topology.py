from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .basins import AttractorDescriptor, BasinAffinity
from .models import CausalPath, CausalProvenance, Edge, EdgeKind, MemoryNode


PROVENANCE_FACTORS: dict[CausalProvenance, float] = {
    CausalProvenance.EXECUTION: 1.00,
    CausalProvenance.WORKFLOW: 0.95,
    CausalProvenance.DEPENDENCY: 0.90,
    CausalProvenance.EVENT: 0.90,
    CausalProvenance.INFERRED: 0.60,
    CausalProvenance.HYPOTHESIZED: 0.35,
}

BASIN_EDGE_KINDS = {EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL}


class CausalTopology:
    def __init__(self) -> None:
        self.nodes: dict[str, MemoryNode] = {}
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        self._attractors: dict[str, AttractorDescriptor] = {}

    def add_node(self, node: MemoryNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"node already exists: {node.id}")
        self.nodes[node.id] = node
        if node.is_attractor:
            self._attractors[node.id] = AttractorDescriptor(node_id=node.id, origin="legacy")

    def upsert_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node
        if node.is_attractor and node.id not in self._attractors:
            self._attractors[node.id] = AttractorDescriptor(node_id=node.id, origin="legacy")

    def has_edge(self, edge: Edge) -> bool:
        identity = edge.identity()
        return any(existing.identity() == identity for existing in self._out.get(edge.source, []))

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise KeyError("both edge endpoints must exist before adding an edge")
        if self.has_edge(edge):
            raise ValueError(
                "edge already exists: "
                f"{edge.source}->{edge.target}:{edge.kind.value}:{edge.provenance}"
            )
        self._out[edge.source].append(edge)
        self._in[edge.target].append(edge)

    def outgoing(self, node_id: str, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]:
        edges = self._out.get(node_id, [])
        if kinds is None:
            return list(edges)
        allowed = set(kinds)
        return [edge for edge in edges if edge.kind in allowed]

    def incoming(self, node_id: str, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]:
        edges = self._in.get(node_id, [])
        if kinds is None:
            return list(edges)
        allowed = set(kinds)
        return [edge for edge in edges if edge.kind in allowed]

    def register_attractor(
        self,
        node_id: str,
        *,
        confidence: float = 1.0,
        origin: str = "manual",
        metadata: dict | None = None,
    ) -> AttractorDescriptor:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        descriptor = AttractorDescriptor(
            node_id=node_id,
            confidence=confidence,
            origin=origin,
            metadata=dict(metadata or {}),
        )
        self.nodes[node_id].is_attractor = True
        self._attractors[node_id] = descriptor
        return descriptor

    def unregister_attractor(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        self.nodes[node_id].is_attractor = False
        self._attractors.pop(node_id, None)

    def attractor(self, node_id: str) -> AttractorDescriptor | None:
        return self._attractors.get(node_id)

    def attractor_descriptors(self) -> list[AttractorDescriptor]:
        return [self._attractors[node_id] for node_id in sorted(self._attractors)]

    def basin(self, attractor_id: str, max_hops: int = 8) -> set[str]:
        if attractor_id not in self.nodes:
            raise KeyError(attractor_id)
        if attractor_id not in self._attractors:
            raise ValueError(f"node is not registered as an attractor: {attractor_id}")
        return self.neighborhood(
            attractor_id,
            direction="in",
            kinds=BASIN_EDGE_KINDS,
            max_hops=max_hops,
            include_anchor=True,
        )

    def attractors_for(self, node_id: str, max_hops: int = 8) -> set[str]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        reachable = self.neighborhood(
            node_id,
            direction="out",
            kinds=BASIN_EDGE_KINDS,
            max_hops=max_hops,
            include_anchor=True,
        )
        return set(self._attractors).intersection(reachable)

    def basin_memberships(self, node_id: str, max_hops: int = 8) -> set[str]:
        return self.attractors_for(node_id, max_hops=max_hops)

    def shared_basin_affinity(
        self,
        left_id: str,
        right_id: str,
        *,
        max_hops: int = 8,
    ) -> BasinAffinity:
        shared = tuple(sorted(
            self.attractors_for(left_id, max_hops=max_hops).intersection(
                self.attractors_for(right_id, max_hops=max_hops)
            )
        ))
        if not shared:
            return BasinAffinity(left_id, right_id, (), 0.0)
        confidence = max(self._attractors[node_id].confidence for node_id in shared)
        return BasinAffinity(left_id, right_id, shared, 0.6 * confidence)

    def basin_boundary(self, attractor_id: str, max_hops: int = 8) -> set[str]:
        members = self.basin(attractor_id, max_hops=max_hops)
        boundary: set[str] = set()
        for node_id in members:
            neighbors = [edge.target for edge in self.outgoing(node_id, BASIN_EDGE_KINDS)]
            neighbors.extend(edge.source for edge in self.incoming(node_id, BASIN_EDGE_KINDS))
            if any(neighbor not in members for neighbor in neighbors):
                boundary.add(node_id)
        return boundary

    def neighboring_basins(self, attractor_id: str, max_hops: int = 8) -> set[str]:
        members = self.basin(attractor_id, max_hops=max_hops)
        neighbors: set[str] = set()

        # Overlapping basins are neighbors by shared state.
        for other_id in self._attractors:
            if other_id == attractor_id:
                continue
            if members.intersection(self.basin(other_id, max_hops=max_hops)):
                neighbors.add(other_id)

        # Basins connected across a boundary are also neighbors.
        for node_id in self.basin_boundary(attractor_id, max_hops=max_hops):
            adjacent = [edge.target for edge in self.outgoing(node_id, BASIN_EDGE_KINDS)]
            adjacent.extend(edge.source for edge in self.incoming(node_id, BASIN_EDGE_KINDS))
            for adjacent_id in adjacent:
                if adjacent_id in members:
                    continue
                neighbors.update(self.attractors_for(adjacent_id, max_hops=max_hops))
        neighbors.discard(attractor_id)
        return neighbors

    def neighborhood(
        self,
        node_id: str,
        *,
        direction: str = "both",
        kinds: Iterable[EdgeKind] | None = None,
        max_hops: int = 4,
        include_anchor: bool = False,
    ) -> set[str]:
        distances = self.distances(
            node_id,
            direction=direction,
            kinds=kinds,
            max_hops=max_hops,
        )
        result = set(distances)
        if not include_anchor:
            result.discard(node_id)
        return result

    def distances(
        self,
        node_id: str,
        *,
        direction: str = "both",
        kinds: Iterable[EdgeKind] | None = None,
        max_hops: int = 4,
    ) -> dict[str, int]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be one of: in, out, both")
        if node_id not in self.nodes:
            raise KeyError(node_id)
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        allowed = set(kinds) if kinds is not None else None
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        distances = {node_id: 0}

        while queue:
            current, hops = queue.popleft()
            if hops >= max_hops:
                continue
            edges: list[tuple[Edge, str]] = []
            if direction in {"out", "both"}:
                edges.extend((edge, edge.target) for edge in self._out.get(current, []))
            if direction in {"in", "both"}:
                edges.extend((edge, edge.source) for edge in self._in.get(current, []))

            for edge, neighbor in edges:
                if allowed is not None and edge.kind not in allowed:
                    continue
                candidate_hops = hops + 1
                if neighbor in distances and distances[neighbor] <= candidate_hops:
                    continue
                distances[neighbor] = candidate_hops
                queue.append((neighbor, candidate_hops))
        return distances

    @staticmethod
    def provenance_factor(provenance: CausalProvenance) -> float:
        return PROVENANCE_FACTORS[provenance]

    def causal_paths(
        self,
        anchor_id: str,
        candidate_id: str,
        *,
        direction: str,
        max_hops: int = 4,
    ) -> list[tuple[tuple[str, ...], tuple[Edge, ...], float]]:
        if anchor_id not in self.nodes:
            raise KeyError(anchor_id)
        if candidate_id not in self.nodes:
            raise KeyError(candidate_id)
        if direction not in {"in", "out"}:
            raise ValueError("causal path direction must be in or out")
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if anchor_id == candidate_id:
            return [((anchor_id,), (), 1.0)]
        if max_hops == 0:
            return []

        results: list[tuple[tuple[str, ...], tuple[Edge, ...], float]] = []
        stack: list[tuple[str, tuple[str, ...], tuple[Edge, ...], float]] = [
            (anchor_id, (anchor_id,), (), 1.0)
        ]
        while stack:
            current, nodes, edges_so_far, confidence = stack.pop()
            if len(edges_so_far) >= max_hops:
                continue
            edges = self._in.get(current, []) if direction == "in" else self._out.get(current, [])
            for edge in edges:
                if edge.kind is not EdgeKind.CAUSAL or edge.provenance is None:
                    continue
                neighbor = edge.source if direction == "in" else edge.target
                if neighbor in nodes:
                    continue
                factor = edge.confidence * min(1.0, edge.weight) * self.provenance_factor(edge.provenance)
                next_confidence = confidence * factor
                next_nodes = nodes + (neighbor,)
                next_edges = edges_so_far + (edge,)
                if neighbor == candidate_id:
                    results.append((next_nodes, next_edges, next_confidence))
                    continue
                if len(next_edges) < max_hops:
                    stack.append((neighbor, next_nodes, next_edges, next_confidence))

        results.sort(key=lambda item: (-item[2], len(item[1]), item[0]))
        return results

    def causal_path_evidence(
        self,
        anchor_id: str,
        candidate_id: str,
        *,
        direction: str,
        max_hops: int = 4,
    ) -> CausalPath | None:
        paths = self.causal_paths(
            anchor_id,
            candidate_id,
            direction=direction,
            max_hops=max_hops,
        )
        if not paths:
            return None
        best_nodes, best_edges, best_confidence = paths[0]
        remaining = 1.0
        for _, _, confidence in paths:
            remaining *= 1.0 - max(0.0, min(1.0, confidence))
        aggregate = 1.0 - remaining
        return CausalPath(
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            direction=direction,
            nodes=best_nodes,
            edges=best_edges,
            best_confidence=best_confidence,
            aggregate_confidence=max(0.0, min(1.0, aggregate)),
        )

    def causal_path_confidence(
        self,
        anchor_id: str,
        candidate_id: str,
        *,
        direction: str,
        max_hops: int = 4,
    ) -> tuple[int | None, float]:
        path = self.causal_path_evidence(
            anchor_id,
            candidate_id,
            direction=direction,
            max_hops=max_hops,
        )
        if path is None:
            return None, 0.0
        return path.hops, path.aggregate_confidence
