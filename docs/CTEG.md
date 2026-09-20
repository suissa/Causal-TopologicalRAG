# CTEG — Causal-Temporal Experiential Graph

## Definition

The mandatory memory substrate of CT-RAG is the **Causal-Temporal Experiential Graph (CTEG)**.

> **CTEG stores what happened.**

CTEG is a heterogeneous experiential graph in which causal authority, temporal ordering, behavioral continuity, semantic similarity, evidence provenance and navigational influence remain distinct dimensions.

The evidence/provenance inventory below names **sources that can justify or contextualize graph state**. It does **not** mean that every Event, Trace, Log, Metric, Config or Code artifact becomes a vertex in (V). Typed Evidence Anchors are projection inputs/evidence pointers; vertex materialization is a separate projector decision.

```text
CT-RAG
│
├── Structured Experiential Memory
│
└── Causal-Temporal Experiential Graph (CTEG)
    │
    ├── CAUSAL
    ├── TEMPORAL
    ├── BEHAVIORAL
    ├── SEMANTIC
    │
    └── evidence / provenance
         ├── Event
         ├── Trace
         ├── Log
         ├── Metric
         ├── Config
         └── Code
```

The term **CTEG Database** refers to a persistent database implementation of this graph contract. The current in-memory reference implementation remains `CausalTopology`; storage adapters may implement the same semantics without changing retrieval behavior.

## Relation semantics

### CAUSAL

A causal edge requires independent causal authority. Examples include explicit runtime `causation_id`, declared workflow dependency, dependency provenance supplied by an authoritative runtime, or explicitly marked inferred/hypothesized relations whose weaker epistemic status remains visible.

A temporal, trace, semantic or behavioral relation must never be silently upgraded to CAUSAL.

### TEMPORAL

Temporal edges answer **when / before / after / within which scope?** They can be scoped to execution, actor or aggregate, deployment, incident window, or global observed time.

CT-RAG distinguishes:

```text
event_time
!=
observed_at
```

`event_time` is when the source says the event happened. `observed_at` is when CT-RAG observed/ingested it. Arrival order therefore does not redefine event-time order or causal authority.

### BEHAVIORAL

Behavioral edges preserve execution continuity and repeated action/trajectory structure. They describe how execution progressed, not why one state caused another.

### SEMANTIC

Semantic similarity is a retrieval signal. It may be represented as a materialized relation or computed on demand by embeddings/lexical retrieval. Semantic similarity never establishes causal authority.

## Trace as the execution spine

A trace is not a causal oracle.

> **A trace provides the execution spine for reconstructing a scenario: the path of a payload across components, together with the events, logs, metrics, configuration and other evidence observed around that execution in a given temporal window. Causal authority remains a separate concern and requires explicit provenance.**

Conceptually:

```text
                    TRACE
                      │
              execution spine
                      │
        ┌─────────────┼─────────────┐
        │             │             │
      Event          Log          Metric
        │             │             │
      Config         Code        Tool Call
        │             │             │
        └─────────────┼─────────────┘
                      │
             Temporal Reconstruction
                      │
        ┌─────────────┴─────────────┐
        │                           │
 Payload path                System scenario
 "where did it go?"          "what was happening?"
```

A trace reconstructs two complementary views:

1. **Payload path reconstruction** — which components, operations and spans the payload traversed.
2. **System-scenario reconstruction** — surrounding events, logs, metrics, config and tool/agent activity inside the relevant temporal window.

Trace parent/child hierarchy remains artifact-local or behavioral structure unless separate causal provenance exists.

## Dynamic Terrain

Dynamic Terrain is an overlay over CTEG, not the authoritative database itself.

```text
Dynamic Terrain
────────────────────
reinforcement
erosion
surprise
basin
attractor
basin drift
```

It changes **navigational influence**, never historical truth.

### Reinforcement

A repeatedly observed transition can gain influence.

### Erosion

Influence can decay with elapsed time and disuse without deleting the underlying historical edge.

### Surprise

Surprise answers:

> **How unexpected was this observed transition or outcome relative to the experiential history available at this point?**

The default transition residual is based on an empirical outgoing probability:

`Surprise(e) = 1 - P(e | source(e), kind(e))`

with smoothing in the implementation. A rare transition may therefore receive a larger surprise value than a dominant repeated path.

Important separation:

```text
frequency
    = how often it happened

terrain influence
    = how strongly it currently influences navigation

surprise
    = how unexpected the new observation was

causal authority
    = why the system is allowed to assert A caused B
```

Surprise may modify attention or reinforcement strength. It must never manufacture a CAUSAL edge.

### Basin and attractor

Basins describe regions whose observed trajectories converge toward an attractor. Attractors may represent recovery, failure, compensation loops, human intervention or other recurrent terminal behavior.

### Basin drift

Basin drift measures change in basin membership/structure between snapshots. It describes topological evolution; it is not itself an intervention-effect estimate.

## Core invariants

```text
trace parent-child       != causal relation
temporal adjacency       != causal relation
semantic similarity      != causal relation
behavioral continuity    != causal relation
surprise                 != causal relation
terrain influence        != historical truth
```

Only evidence with appropriate causal provenance can enter the authoritative causal relation.

## Required vs optional memory

CTEG is the **required** structured experiential memory substrate of CT-RAG.

A cognitive memory layer is intentionally optional and separate. See [COGNITIVE_GRAPH.md](COGNITIVE_GRAPH.md).
