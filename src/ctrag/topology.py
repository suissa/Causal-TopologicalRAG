from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .models import Edge, EdgeKind, MemoryNode


class CausalTopology:
    def __init__(self) -> None:
        self.nodes: dict[str, MemoryNode] = {}
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)

    def add_node(self, node: MemoryNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"node already exists: {node.id}")
        self.nodes[node.id] = node

    def upsert_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node

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

    def register_attractor(self, node_id: str) -> None:
        self.nodes[node_id].is_attractor = True

    def basin(self, attractor_id: str, max_hops: int = 8) -> set[str]:
        if attractor_id not in self.nodes:
            raise KeyError(attractor_id)
        if not self.nodes[attractor_id].is_attractor:
            raise ValueError(f"node is not registered as an attractor: {attractor_id}")
        return self.neighborhood(
            attractor_id,
            direction="in",
            kinds={EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL},
            max_hops=max_hops,
            include_anchor=True,
        )

    def attractors_for(self, node_id: str, max_hops: int = 8) -> set[str]:
        reachable = self.neighborhood(
            node_id,
            direction="out",
            kinds={EdgeKind.CAUSAL, EdgeKind.BEHAVIORAL},
            max_hops=max_hops,
            include_anchor=True,
        )
        return {candidate for candidate in reachable if self.nodes[candidate].is_attractor}

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

    def causal_path_confidence(
        self,
        anchor_id: str,
        candidate_id: str,
        *,
        direction: str,
        max_hops: int = 4,
    ) -> tuple[int | None, float]:
        """Return shortest causal hops and best confidence product at that depth."""
        if anchor_id not in self.nodes:
            raise KeyError(anchor_id)
        if candidate_id not in self.nodes:
            raise KeyError(candidate_id)
        if anchor_id == candidate_id:
            return 0, 1.0
        if direction not in {"in", "out"}:
            raise ValueError("causal path direction must be in or out")
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")

        queue: deque[tuple[str, int, float]] = deque([(anchor_id, 0, 1.0)])
        best_seen: dict[tuple[str, int], float] = {(anchor_id, 0): 1.0}
        best_target_hops: int | None = None
        best_target_confidence = 0.0

        while queue:
            current, hops, confidence = queue.popleft()
            if hops >= max_hops:
                continue
            edges = self._in.get(current, []) if direction == "in" else self._out.get(current, [])
            for edge in edges:
                if edge.kind is not EdgeKind.CAUSAL:
                    continue
                neighbor = edge.source if direction == "in" else edge.target
                next_hops = hops + 1
                next_confidence = confidence * edge.confidence * edge.weight
                state = (neighbor, next_hops)
                if next_confidence <= best_seen.get(state, -1.0):
                    continue
                best_seen[state] = next_confidence
                if neighbor == candidate_id:
                    if best_target_hops is None or next_hops < best_target_hops:
                        best_target_hops = next_hops
                        best_target_confidence = next_confidence
                    elif next_hops == best_target_hops:
                        best_target_confidence = max(best_target_confidence, next_confidence)
                if best_target_hops is None or next_hops < best_target_hops:
                    queue.append((neighbor, next_hops, next_confidence))

        return best_target_hops, min(1.0, best_target_confidence)
