# Basins and attractors

CT-RAG treats basins and attractors as queryable topology, not presentation metadata.

## Attractor descriptors

`register_attractor()` creates an `AttractorDescriptor` containing:

- `node_id`;
- confidence in `[0, 1]`;
- `origin` (for example `manual`, `legacy`, or later `discovered`);
- arbitrary metadata.

The legacy `MemoryNode.is_attractor` flag remains synchronized for compatibility, while descriptors provide the richer research contract required for persistence and empirical discovery.

## Basin definition

For an attractor `A`, its current basin is the reverse-reachable set using only causal and behavioral edges up to a configured hop budget:

```text
B(A) = { x | x reaches A through causal/behavioral topology }
```

Temporal adjacency alone does not create basin membership.

APIs:

```python
topology.basin(attractor_id)
topology.basin_memberships(node_id)
topology.attractor_descriptors()
```

A state may belong to multiple basins in a branching graph. This is intentional: a pre-divergence state can still reach several outcomes.

## Explainable shared-basin affinity

`shared_basin_affinity(left, right)` returns `BasinAffinity` with the exact shared attractor IDs and the scalar score used as a topological prior. The current compatibility score is:

```text
0.6 × max(shared attractor confidence)
```

With default confidence `1.0`, this preserves the `0.6` basin prior used by the original benchmark.

## Boundaries and neighboring basins

`basin_boundary(A)` returns basin members touching a causal/behavioral edge to a state outside `B(A)`.

`neighboring_basins(A)` includes basins that overlap `B(A)` or connect across one of those boundary edges. This makes branch relationships inspectable without merging the basins into one cluster.

These definitions are deterministic for a fixed topology and hop budget and remain finite in cyclic graphs because reachability uses visited-node distances.
