# Temporal Topology in CT-RAG

CT-RAG treats time as a first-class retrieval dimension, but **not as a substitute for causality**.

The memory topology is modeled as a heterogeneous directed graph:

[
G=(V,E_s,E_c,E_t,E_b)
]

where `E_t` contains temporal-order relations that remain distinct from causal (`E_c`), semantic (`E_s`) and behavioral/execution (`E_b`) relations.

## Core invariant

```text
A happened before B
        !=
A caused B
```

Temporal adjacency can support navigation, ordering and hypothesis generation, but never promotes itself to causal evidence.

## What is already implemented

The current implementation already contains temporal semantics in several places:

- `MemoryNode.timestamp` records when a memory/event occurred.
- `EdgeKind.TEMPORAL` represents explicit observed ordering.
- `EventProjector` can project temporal relations between events while keeping them separate from `CAUSAL` edges.
- `CTRetriever._temporal_scores()` contributes temporal relevance to retrieval.
- Topological expansion can traverse `CAUSAL`, `BEHAVIORAL` and `TEMPORAL` relations.
- The ranking model already has an independent temporal component.
- External-validity experiments can use temporal topology without inventing causal edges from sequence alone.

## Temporal distance

For memories `x_i` and `x_j`, CT-RAG distinguishes at least two notions of temporal distance:

[
d_t^{clock}(x_i,x_j)=|t_i-t_j|
]

and a graph-relative temporal distance:

[
d_t^{hop}(x_i,x_j)=	ext{minimum number of TEMPORAL edges connecting them}
]

These answer different questions.

Clock distance answers:

> How far apart in time did these observations occur?

Temporal-hop distance answers:

> How far apart are they inside the observed sequence/topology?

Neither implies causality.

## Temporal direction

Time is directional.

For an anchor `a`, retrieval can distinguish:

[
Past(a)=\{v:t_v<t_a\}
]

from

[
Future(a)=\{v:t_v>t_a\}
]

This matters for query modes:

- `WHY` normally prefers relevant past evidence and causal ancestors.
- `WHAT_NEXT` normally prefers observed descendants and later states.
- `RECOVERY` can search historical trajectories that start near a failure and later converge toward recovery.
- `COUNTERFACTUAL` can compare observed branches after a common historical state without treating the branch comparison itself as causal proof.

## Temporal relevance

The retrieval score can include an independent temporal prior:

[
Score(v|q,a)=
\alpha S_{dense}+
\beta S_{lexical}+
\gamma P_{topological}+
\delta P_{causal}+
\epsilon P_{temporal}+
\zeta P_{behavioral}
]

A simple recency prior can be expressed as:

[
P_{temporal}(v)=e^{-\lambda |t_{query}-t_v|}
]

but CT-RAG does **not** require recency to always be preferred. Query intent can change the temporal policy.

For example:

- "what happened immediately before this error?" favors local past proximity;
- "has this exact recovery ever worked?" may prefer older but structurally equivalent evidence;
- "what changed in the last hour?" imposes a hard temporal window;
- "why is this state recurring every Monday?" requires periodic/longitudinal evidence rather than simple recency.

## Temporal topology versus temporal causality

A temporal chain:

```text
A --TEMPORAL--> B --TEMPORAL--> C
```

supports the statement:

```text
A occurred before B, and B before C.
```

It does not support:

```text
A caused B caused C.
```

A causal chain requires independent causal evidence:

```text
A --CAUSAL(provenance=event)--> B
B --CAUSAL(provenance=execution)--> C
```

The same pair of nodes may legitimately have both temporal and causal edges.

## Time and the Dynamic Terrain

CT-RAG also has a second temporal property: **the topology itself evolves through experience**.

The historical graph remains authoritative and immutable in meaning, while `DynamicTerrain` can change navigational influence over time.

Conceptually:

[
w_{ij}(t+1)=w_{ij}(t)+\eta
]

for reinforced observed transitions, and:

[
w_{ij}(t+\Delta t)=w_{ij}(t)e^{-\mu\Delta t}
]

for navigational erosion.

This creates an important separation:

```text
historical truth
    !=
current navigational influence
```

An old event can remain permanently preserved while becoming less influential for retrieval.

## Basin drift is temporal structure

Basins and attractors need not be static.

Given basin snapshots `B_t(A)` and `B_{t+1}(A)`, CT-RAG can measure structural drift over time:

[
Drift(B_t,B_{t+1})
]

This allows questions such as:

- Is the system converging toward a different failure mode than last month?
- Did a healing change reduce the size of a failure basin?
- Has a formerly rare recovery path become a dominant attractor?
- Did a configuration change reshape the execution terrain?

This is more than timestamp filtering: it treats **changes in topology through time** as evidence.

## Temporal layers we should distinguish

For production/event-sourced systems, CT-RAG should eventually distinguish at least:

1. **event time** — when the domain event actually occurred;
2. **ingest time** — when CT-RAG observed/indexed it;
3. **processing time** — when a projection/retrieval computation ran;
4. **valid time** — when a fact/state was valid in the modeled domain;
5. **system time** — when that fact/version existed in the storage model.

The current implementation primarily uses event timestamps. Full bitemporal semantics are a planned extension, not a current claim.

## Planned temporal extensions

The temporal model should be extended with:

- explicit event-time vs ingest-time fields;
- time-window filters;
- interval-valued states;
- configurable temporal decay policies;
- temporal path constraints;
- periodicity/seasonality descriptors;
- bitemporal projections where domains require them;
- temporal consistency checks for out-of-order ingestion;
- time-aware basin/attractor snapshots;
- longitudinal basin-drift benchmarks.

These extensions must preserve the core invariant that temporal proximity is evidence about **order and time**, not automatic evidence of **cause**.

## Mental model

```text
Semantic:
    what looks like this?

Temporal:
    when did this happen, and what was before/after it?

Causal:
    what produced this state?

Behavioral:
    which execution/trajectory does this belong to?

Topological:
    where is this inside the navigable terrain?

Dynamic terrain:
    how has repeated experience changed the importance of paths over time?
```

Time is therefore not an annotation attached to CT-RAG. It is one of the independent dimensions that define the navigable memory terrain.
