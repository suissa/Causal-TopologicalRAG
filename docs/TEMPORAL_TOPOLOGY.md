# Temporal Topology in CT-RAG

CT-RAG treats time as a first-class retrieval dimension, but **not as a substitute for causality**.

The memory topology is modeled as a heterogeneous directed graph:

$$
G=(V,E_s,E_c,E_t,E_b)
$$

where `E_t` contains temporal-order relations that remain distinct from causal (`E_c`), semantic (`E_s`) and behavioral/execution (`E_b`) relations. The direction of a temporal edge is meaningful only under its declared ordering contract; it is not causal evidence.

## Core invariant

```text
A happened before B
        !=
A caused B
```

Temporal adjacency can support navigation, ordering and hypothesis generation, but never promotes itself to causal evidence.

## What is already implemented

The current implementation already contains temporal semantics in several places:

- `MemoryNode.timestamp` carries the current single event timestamp; it is not an ingest-time or bitemporal record.
- `EdgeKind.TEMPORAL` represents observed sequence under a declared scope.
- `EventProjector` projects execution-local temporal relations in projection/ingest order while keeping them separate from `CAUSAL` edges.
- `CTRetriever._temporal_scores()` contributes temporal relevance to retrieval.
- Topological expansion can traverse `CAUSAL`, `BEHAVIORAL` and `TEMPORAL` relations.
- The ranking model already has an independent temporal component.
- External-validity experiments can use temporal topology without inventing causal edges from sequence alone.

## Temporal distance and consistency

For memories `x_i` and `x_j`, CT-RAG distinguishes at least two notions of temporal distance:

$$
d_t^{clock}(x_i,x_j)=|t_i-t_j|
$$

and a graph-relative temporal distance:

$$
d_t^{hop}(x_i,x_j)=\text{minimum number of TEMPORAL edges connecting them}
$$

These answer different questions.

Clock distance answers:

> How far apart in time did these observations occur?

Temporal-hop distance answers:

> How far apart are they inside the observed sequence/topology?

Neither implies causality.


### Temporal Consistency Window

Temporal order can be necessary for a causal hypothesis without being sufficient for one. CT-RAG can therefore apply a **Temporal Consistency Window** to reject paths whose observed clock separation is incompatible with the expected system dynamics.

For a candidate path \(P=(v_0,\ldots,v_n)\), define:

$$
C_{time}(P;\tau)=1 \iff \max_i d_t^{clock}(v_i,v_{i+1}) \le \tau
$$

where \(\tau\) is a domain- or operation-specific bound such as an expected timeout, lease duration, retry interval or SLA-derived window.

This filter does **not** create causality. It only rejects temporally inconsistent candidates that would otherwise survive graph-hop filtering.

The bound must be sourced from system/domain configuration or a declared policy and persisted with the retrieval artifact; it must not be tuned post hoc on the evaluated incident.

## Temporal direction

Time is directional.

For an anchor `a`, retrieval can distinguish:

$$
Past(a)=\{v:t_v<t_a\}
$$

from

$$
Future(a)=\{v:t_v>t_a\}
$$

This matters for query modes:

- `WHY` normally prefers relevant past evidence and causal ancestors.
- `WHAT_NEXT` normally prefers observed descendants and later states.
- `RECOVERY` can search historical trajectories that start near a failure and later converge toward recovery.
- `COUNTERFACTUAL` can compare observed branches after a common historical state without treating the branch comparison itself as causal proof.

## Temporal relevance

The retrieval score can include an independent temporal prior:

$$
Score(v|q,a)=
\alpha S_{dense}+
\beta S_{lexical}+
\gamma P_{topological}+
\delta P_{causal}+
\epsilon P_{temporal}+
\zeta P_{behavioral}
$$

A simple recency prior can be expressed as:

$$
P_{temporal}(v)=e^{-\lambda |t_{query}-t_v|}
$$

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


## Temporal scope hierarchy

The reference projector currently creates execution-local temporal edges between consecutive events sharing one `execution_id`, in the order in which `EventProjector.ingest()` receives them. Their direction records projection/ingest order, not normalized event-time order: a late event may therefore point forward in the projected sequence while carrying an earlier event timestamp. Event-time ordering is a planned, explicit contract rather than an implicit property of the current edge direction.

Production systems also require explicit temporal relations across executions.

Recommended temporal scopes are:

```text
execution
actor_or_aggregate
deployment
incident_window
global_observed
```

Examples:

```text
Deploy.v42 --TEMPORAL(scope=deployment)--> CheckoutExecution.991
ConfigChanged --TEMPORAL(scope=incident_window)--> Incident.2026-09-20
```

A cross-execution temporal relation remains temporal only. It never creates a causal edge without independent provenance.

Retrieval/path constraints should declare which temporal scopes are admissible for the query.

## Time and the Dynamic Terrain

CT-RAG also has a second temporal property: **the topology itself evolves through experience**.

The historical graph remains authoritative and immutable in meaning, while `DynamicTerrain` can change navigational influence over time.

Conceptually:

$$
w_{ij}(t+1)=w_{ij}(t)+\eta
$$

for reinforced observed transitions, and:

$$
w_{ij}(t+\Delta t)=w_{ij}(t)e^{-\mu\Delta t}
$$

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

$$
Drift(B_t,B_{t+1})
$$

The current `DynamicTerrain.basin_drift()` is a Jaccard-distance comparison of basin memberships. It measures **structural membership drift** only; it does not measure transition volume, outcome probability, or whether a path became dominant.

This already supports questions such as:

- Did a healing change the membership/shape of a failure basin?
- Did a configuration change reshape the execution terrain?

Questions such as “has a formerly rare recovery path become dominant?” require a future longitudinal, support-weighted measure over timestamped snapshots. Such a snapshot must record `observed_at`, window policy, topology/terrain version, and transition support so comparisons remain reproducible.

**Temporal Drift Detection Accuracy** is therefore a planned benchmark metric, not a reported result: against a labelled regime change it should report precision, recall, F1, false-alert rate, and lead time under a predeclared window and threshold policy.

This is more than timestamp filtering: it treats **changes in topology through time** as evidence, while preserving the distinction between structural change and flow change.

## Temporal layers we should distinguish

For production/event-sourced systems, CT-RAG should eventually distinguish at least:

1. **event time** — when the domain event actually occurred;
2. **ingest time** — when CT-RAG observed/indexed it;
3. **processing time** — when a projection/retrieval computation ran;
4. **valid time** — when a fact/state was valid in the modeled domain;
5. **system time** — when that fact/version existed in the storage model.

The current implementation primarily uses event timestamps. Full bitemporal semantics are a planned extension, not a current claim.


## Point-in-time reconstruction and bitemporal projection

A production post-mortem often needs a stronger question than “what do we know now about what happened then?” It may need:

> What topology and navigational influence were available to the agent at decision time \(t_0\)?

CT-RAG should therefore support point-in-time projection over both authoritative history and the non-authoritative terrain:

```text
Topology(as_of = t0)
Terrain(as_of = t0)
Basins(as_of = t0)
Attractors(as_of = t0)
```

This requires separating at least valid/event time from system/ingest time. A later correction may change what the system knows today without retroactively changing what the agent could have known at \(t_0\).

The intended post-mortem invariant is:

```text
knowledge available at decision time
!=
knowledge reconstructed with today's database state
```

Full historical terrain snapshots are planned work; the current implementation does not yet claim bitemporal terrain reconstruction.

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
- time-aware basin/attractor snapshots with `observed_at`, window policy and topology/terrain version;
- support-weighted longitudinal basin-drift benchmarks that distinguish structural membership drift from transition-flow drift;
- Temporal Drift Detection Accuracy evaluation against labelled regime changes.

These extensions must preserve the core invariant that temporal proximity is evidence about **order and time**, not automatic evidence of **cause**.


## Clock versus hop example

```text
10:00:00            10:00:01                              17:00:00
   A  --TEMPORAL-->    B   --TEMPORAL-->                     C

clock(A,B) = 1 second
clock(B,C) = 6h 59m 59s
clock(A,C) = 7 hours

temporal-hop(A,B) = 1
temporal-hop(B,C) = 1
temporal-hop(A,C) = 2
```

The same topology supports different retrieval questions:

```text
"what happened chronologically after A?"
    -> temporal scope + clock policy + TEMPORAL traversal

"what was triggered by A?"
    -> CAUSAL traversal only; temporal hops can constrain plausibility but cannot promote an edge
```

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
