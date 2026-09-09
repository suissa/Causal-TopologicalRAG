# Core model contracts

This document defines the public invariants for the CT-RAG model layer.

## `MemoryNode`

`MemoryNode` is a stable memory/state identity. `id` must be a non-empty string and `timestamp` must be timezone-aware. Embedding values, when supplied, must be finite.

`MemoryNode.to_dict()` and `MemoryNode.from_dict()` are the canonical JSON-compatible round-trip contract. Timestamps are serialized with ISO-8601 offsets and embeddings as arrays.

## `EdgeKind`

Relations are intentionally distinct:

- `semantic`: similarity/association only;
- `causal`: an asserted causal relation with provenance;
- `temporal`: ordering only;
- `behavioral`: shared execution/behavioral flow.

A temporal or behavioral edge is never promoted to causal merely because it is adjacent in time.

## `CausalProvenance`

Causal edges require exactly one provenance class:

- `execution`;
- `dependency`;
- `workflow`;
- `event`;
- `inferred`;
- `hypothesized`.

The first four can represent observed/system evidence when the producer has that authority. `inferred` and `hypothesized` remain explicitly distinguishable and must not be presented as observed execution causality.

## `EdgeEvidence`

`EdgeEvidence` attaches stable evidence references to a causal edge. It contains:

- a non-empty evidence `id`;
- optional `source`;
- open-ended JSON-compatible `metadata`.

Evidence IDs must be unique within an edge.

## `Edge`

An edge has structural identity:

```text
(source, target, kind, provenance)
```

The topology rejects a second edge with the same structural identity. A causal and temporal edge between the same nodes are still different edges because `kind` is part of the identity.

Rules:

- `confidence` is finite and in `[0, 1]`;
- `weight` is finite and non-negative;
- causal edges require `provenance`;
- non-causal edges cannot carry causal provenance, causal evidence or provenance metadata;
- evidence/provenance metadata can be extended without changing edge identity.

`Edge.to_dict()` / `Edge.from_dict()` preserve kind, provenance, confidence, weight, evidence and provenance metadata.

## `RetrievalWeights`

Every component weight must be finite and non-negative. The retriever additionally requires the active set of weights to sum to 1 before ranking.

## Topology identity behavior

`CausalTopology.add_node()` rejects duplicate node IDs. `CausalTopology.add_edge()` rejects duplicate structural edge identities and requires both endpoints to exist first.

These fail-fast rules prevent replay or ingestion bugs from silently creating multiple copies of the same semantic relation. Event replay idempotency is handled at the Event Sourcing projection boundary rather than by weakening topology identity.
