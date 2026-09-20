# CT-RAG Architecture

## Architectural position

CT-RAG is a Structured Experiential Memory layer. It does not require all evidence to be flattened into text before retrieval.

The ingestion/retrieval architecture is:

```text
Raw Evidence
  |
  |  text / table / config / trace / metric / event / log / code / payload
  v
+---------------------------+
| Evidence Shape Router     |
| deterministic-first       |
| semantic fallback         |
+-------------+-------------+
              |
              v
+---------------------------+
| Shape-specific Adapters   |
| config / trace / metric   |
| table / event / log / ... |
+-------------+-------------+
              |
              v
+---------------------------+
| Typed Evidence Anchors    |
| structure + provenance    |
| preserve[] + locator      |
+-------------+-------------+
              |
              v
+---------------------------+
| Causal / Event Projector  |
| E_c / E_t / E_b           |
| provenance policies       |
+-------------+-------------+
              |
              v
+---------------------------+
| Experiential Topology     |
| trajectories / basins     |
| attractors                |
+-------------+-------------+
              |
              v
+---------------------------+
| Terrain Builder           |
| reinforcement / erosion   |
| surprise / drift          |
+-------------+-------------+
              |
              v
+---------------------------+
| Staged CT-RAG Retrieval   |
| anchor -> expand ->       |
| traverse -> rerank        |
+---------------------------+
```

## Evidence Shape Router precedes projection

The Evidence Shape Router is explicitly upstream of the causal/event projector.

Its responsibility is to preserve the native structure required to interpret evidence correctly before experiential relationships are created.

Examples:

```text
CONFIG_TREE
  inventory.reservation.timeout_ms

TRACE
  trace_id / span_id / parent_span_id

METRIC_SERIES
  series / timestamp / window / labels
```

These structures are converted into Typed Evidence Anchors rather than anonymous text chunks.

See [`EVIDENCE_SHAPE_ROUTER.md`](EVIDENCE_SHAPE_ROUTER.md).

## Typed Evidence Anchors are the projection boundary

The projection boundary is not raw evidence.

```text
raw evidence
    -> shape detection
    -> adapter
    -> TypedEvidenceAnchor
    -> projection
```

A Typed Evidence Anchor is **not automatically a vertex in** (V). It is a typed projection input and evidence reference. The projector decides whether an experiential vertex should be created from it, whether it should only annotate/prove an existing vertex or edge, or whether it should remain external evidence addressed by provenance.

This distinction is deliberate:

```text
TypedEvidenceAnchor
    !=
Experiential vertex

Log / Metric / Config / Code evidence
    !=
automatic node in V
```

For example, a log line or metric sample may support the reconstruction of an execution state or an edge without becoming its own persistent experiential node. Vertices represent projected experiential entities/states/events chosen by the graph model; anchors preserve source structure, location and provenance needed to justify that projection.

A Typed Evidence Anchor carries the normalized evidence unit plus the structural information the adapter was required to preserve:

```text
id
shape
source identity
source URI
observed/event time
structural locator
preserved fields
local relations
retrieval features
provenance / lineage
classifier + adapter version
```

The projector may use those typed fields to construct experiential edges, but only according to explicit provenance rules.

## Projection semantics

The projector maps evidence from Typed Evidence Anchors into the experiential graph:

```text
G = (V, E_s, E_c, E_t, E_b)
```

Projection is selective. An anchor may produce a vertex, contribute attributes/evidence to an existing vertex or edge, or remain only as an external evidence pointer. Therefore the evidence inventory is not isomorphic to (V).

where:

- `E_s` represents semantic relationships;
- `E_c` represents causal relationships with explicit provenance;
- `E_t` represents temporal relationships with explicit scope;
- `E_b` represents behavioral/execution relationships.

Critical invariant:

```text
intrinsic artifact relation
        !=
observed causal relation
```

For example:

```text
trace parent -> child
```

is useful execution structure, but must not automatically become:

```text
parent --CAUSAL--> child
```

without independent causal authority.

## Terrain is built over typed/projection identities, not raw evidence

`DynamicTerrain` is a non-authoritative navigational overlay.

It must be constructed over edges/nodes whose identity derives from Typed Evidence Anchors and the experiential projection. Raw files, log strings, metric arrays or arbitrary chunks are not direct terrain entities.

Conceptually:

```text
RawEvidence
    X
    |
    |  not directly
    v
DynamicTerrain

RawEvidence
    -> TypedEvidenceAnchor
    -> Experiential Node/Edge
    -> DynamicTerrain
```

This ensures that reinforcement, erosion, protection and basin drift operate on evidence with stable provenance and structural identity.

Terrain never rewrites:

- the raw source;
- the Typed Evidence Anchor;
- authoritative causal provenance;
- historical event identity.

It modifies only navigational influence.

## Query path

At query time, evidence-shape-aware retrieval and experiential retrieval compose in stages:

```text
Query / Intent
      |
      v
Evidence Requirements
      |
      v
Evidence Shape Router
      |
      v
local shape-aware retrieval
      |
      v
Typed Evidence Anchors
      |
      v
CT-RAG anchor selection
      |
      v
basin/topology expansion
      |
      v
directed causal/temporal/behavioral traversal
      |
      v
terrain-aware reranking
```

## Diagnostic benchmark priority

The first Evidence Shape Router implementation should focus on:

```text
CONFIG_TREE
TRACE
METRIC_SERIES
```

because these three shapes are directly useful for the cross-artifact diagnostic benchmark:

```text
configuration threshold
      +
trace latency
      +
metric drift/anomaly
      |
      v
CT-RAG correlation
      |
      v
CodeManager hypothesis
```

TABLE, EVENT, LOG, CODE and STRUCTURED_PAYLOAD remain part of the public shape contract and can receive dedicated adapters incrementally.

## Component boundaries

### Evidence Shape Router

Answers: **How should this artifact be read?**

### Evidence Adapter

Answers: **Which structural units and fields must survive retrieval?**

### Causal/Event Projector

Answers: **Which experiential relations can be asserted from the available provenance?**

### Dynamic Terrain

Answers: **How should preserved experiential paths influence navigation now?**

### CT-RAG Retriever

Answers: **Which experiential evidence should be navigated for this intent?**

### CodeManager or downstream reasoner

Answers: **Which hypothesis best explains the retrieved evidence jointly?**

## Non-goals

The architecture does not assert that:

- every artifact requires an LLM classifier;
- artifact structure is causal structure;
- trace hierarchy is automatically causality;
- terrain influence is historical truth;
- shape-aware retrieval is part of the core CT-RAG novelty until the factorial ablation establishes its independent contribution.

## Implementation status

- Evidence Shape Router specification: defined.
- Typed Evidence Anchor contract: defined conceptually.
- Generic external trace adapter: present.
- Full `router.py`: tracked in issue #58.
- Initial Config/Trace/Metric adapters: tracked in issue #58.
- Factorial shape-aware × CT-RAG evaluation: tracked in issue #56.
## CTEG as the mandatory memory substrate

The experiential topology above is formally the **Causal-Temporal Experiential Graph (CTEG)**.

> **CTEG stores what happened.**

Its relation families remain epistemically separate:

```text
CAUSAL      -> authority from causal provenance
TEMPORAL    -> scoped order / event-time relation
BEHAVIORAL  -> execution continuity / trajectory
SEMANTIC    -> similarity/retrieval relation
```

The system must never use one relation family as an implicit substitute for another.

### Trace reconstruction boundary

A trace supplies an **execution spine**. It can align the path of a payload with surrounding events, logs, metrics, configuration, code/tool activity and agent actions inside a temporal window.

```text
Trace
  -> payload path reconstruction
  -> system-scenario reconstruction
  -> temporal/behavioral context
  -X-> causal authority
```

Trace parent/child structure therefore remains local/behavioral unless independent causal provenance is supplied.

### Dynamic Terrain boundary

Dynamic Terrain is not the authoritative graph. It overlays CTEG with current navigational influence:

```text
reinforcement
erosion
surprise
basin
attractor
basin drift
```

`surprise` measures how unexpected an observed transition/outcome is relative to experiential history. It may affect attention or reinforcement, but it cannot create causal authority.

### Optional cognitive memory

The Cognitive Graph is intentionally separate and optional:

```text
CTEG Database       REQUIRED   -> observed experience
Cognitive Graph     OPTIONAL   -> belief / goal / hypothesis / decision / plan / prediction
```

A cognitive node may reference CTEG evidence, but it must never contaminate observed historical truth simply because the agent believed or predicted something.

See [`CTEG.md`](CTEG.md) and [`COGNITIVE_GRAPH.md`](COGNITIVE_GRAPH.md).
