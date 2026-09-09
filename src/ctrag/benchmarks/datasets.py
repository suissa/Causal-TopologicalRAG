"""Synthetic ground truth is authored from trace templates, never retrieval output."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from ctrag.models import CausalProvenance, Edge, EdgeKind, MemoryNode, QueryMode
from ctrag.topology import CausalTopology

GENERATOR_VERSION = 1


@dataclass
class Query:
    id: str
    text: str
    mode: QueryMode
    anchor: str
    relevance: dict[str, int]
    causal_nodes: list[str]
    causal_paths: list[list[str]]
    causal_distances: dict[str, int]
    trajectory: list[str] | None = None
    basin_nodes: list[str] | None = None
    recovery_nodes: list[str] | None = None


@dataclass
class Dataset:
    name: str
    seed: int
    topology: CausalTopology
    queries: list[Query]

    def manifest(self) -> dict:
        return {
            "name": self.name, "seed": self.seed, "generator_version": GENERATOR_VERSION,
            "nodes": [dict(id=n.id, text=n.text, timestamp=n.timestamp.isoformat(),
                           metadata=n.metadata, is_attractor=n.is_attractor)
                      for n in sorted(self.topology.nodes.values(), key=lambda n: n.id)],
            "edges": [asdict(e) for node_id in sorted(self.topology.nodes)
                      for e in sorted(self.topology.outgoing(node_id),
                                      key=lambda e: (e.target, e.kind.value))],
            "queries": [asdict(q) for q in self.queries],
        }


def generate(name: str, seed: int, traces: int = 4) -> Dataset:
    if name not in {"failure_recovery", "branching"}:
        raise ValueError(f"unknown dataset: {name}")
    if traces < 1:
        raise ValueError("traces must be positive")
    rng = random.Random(seed)
    topology = CausalTopology()
    queries = []
    nodes = []
    edges = []
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for trace in range(traces):
        service = rng.choice(["inventory", "payment", "shipping", "identity"])
        incident = f"{service} operation failed"
        if name == "failure_recovery":
            texts = ["request accepted", "stale lease acquired", incident,
                     "release lease and refresh credentials", "retry passed validation",
                     "operation recovered successfully"]
            links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
            attractors = {5}
        else:
            texts = ["request accepted", "evaluate authorization policy",
                     "credentials approved", "operation completed successfully",
                     "credentials rejected", incident]
            links = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
            attractors = {3, 5}
        # Opaque, shuffled IDs keep tie breaks independent of labels and trace order.
        ids = [f"n-{rng.getrandbits(96):024x}" for _ in range(9)]
        texts += [incident + " troubleshooting guide", incident + " archived report",
                  "unrelated heartbeat observed"]
        for index, (node_id, text) in enumerate(zip(ids, texts, strict=True)):
            nodes.append(MemoryNode(
                id=node_id, text=text,
                timestamp=epoch + timedelta(seconds=trace * 100 + index),
                metadata={"execution_id": f"run-{trace}" if index < 6 else f"noise-{trace}-{index}"},
                is_attractor=index in attractors,
            ))
        for source, target in links:
            edges.append(Edge(ids[source], ids[target], EdgeKind.CAUSAL,
                              provenance=CausalProvenance.EXECUTION))
        # A lookalike is temporally adjacent, but is explicitly NOT a cause.
        edges.append(Edge(ids[6], ids[2 if name == "failure_recovery" else 5], EdgeKind.TEMPORAL))

        def query(label, mode, anchor, paths, *, trajectory=None, basin=None, recovery=None):
            relevant = sorted({i for path in paths for i in path if i != anchor})
            distances = {}
            for path in paths:
                for i in path:
                    if i != anchor:
                        distance = abs(path.index(i) - path.index(anchor))
                        distances[ids[i]] = min(distances.get(ids[i], distance), distance)
            queries.append(Query(
                id=f"{trace}-{label}", text=texts[anchor] if mode is QueryMode.WHY else f"{service} recovery next steps",
                mode=mode, anchor=ids[anchor],
                relevance={ids[i]: 2 if distances[ids[i]] == 1 else 1 for i in relevant},
                causal_nodes=[ids[i] for i in relevant],
                causal_paths=[[ids[i] for i in path] for path in paths],
                causal_distances=distances,
                trajectory=[ids[i] for i in trajectory] if trajectory is not None else None,
                basin_nodes=[ids[i] for i in basin] if basin is not None else None,
                recovery_nodes=[ids[i] for i in recovery] if recovery is not None else None,
            ))

        if name == "failure_recovery":
            query("why", QueryMode.WHY, 2, [[0, 1, 2]], trajectory=[0, 1, 2], basin=list(range(6)))
            query("recovery", QueryMode.RECOVERY, 2, [[2, 3, 4, 5]],
                  trajectory=[2, 3, 4, 5], basin=list(range(6)), recovery=[3, 4, 5])
            query("next", QueryMode.WHAT_NEXT, 3, [[3, 4, 5]], trajectory=[3, 4, 5], basin=list(range(6)))
        else:
            query("why-failure", QueryMode.WHY, 5, [[0, 1, 4, 5]],
                  trajectory=[0, 1, 4, 5], basin=[0, 1, 4, 5])
            query("why-success", QueryMode.WHY, 3, [[0, 1, 2, 3]],
                  trajectory=[0, 1, 2, 3], basin=[0, 1, 2, 3])
            query("branch-next", QueryMode.WHAT_NEXT, 1, [[1, 2, 3], [1, 4, 5]])
    rng.shuffle(nodes)
    for node in nodes:
        topology.add_node(node)
    for edge in edges:
        topology.add_edge(edge)
    return Dataset(name, seed, topology, queries)
