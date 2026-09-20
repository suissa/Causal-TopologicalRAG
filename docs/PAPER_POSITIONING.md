# Paper Positioning: CT-RAG as Structured Experiential Memory

## Core positioning

CT-RAG should not be framed as a replacement for Vector RAG, GraphRAG, Topo-RAG or BasinRAG.

Its proposed role is a **Structured Experiential Memory layer for long-lived agents and stateful systems**.

The system retains conventional retrieval as an anchoring mechanism, then adds execution-aware navigation over historical experience:

```text
Vector / lexical retrieval
        ↓
candidate memories
        ↓
Structured Experiential Memory
        ↓
causal / temporal / behavioral navigation
        ↓
basins / attractors / terrain
```

This makes CT-RAG compositional rather than substitutive.

## Abstract-level tagline

> **Retrieval becomes navigation; learning becomes terrain modification.**

The intended meaning is precise:

- *retrieval becomes navigation*: relevant context is selected not only by semantic similarity, but by traversing typed relationships and historical trajectories;
- *learning becomes terrain modification*: repeated or surprising experience updates navigational influence without rewriting the authoritative event history.

This tagline must not be used to claim graph navigation itself as novel; HippoRAG and other graph-memory systems are prior art for navigational retrieval. The CT-RAG claim concerns navigation over **execution-provenance causal/temporal/behavioral memory plus adaptive basin/terrain structure**.

## Proposed abstract framing

A defensible paper abstract should center the problem of long-lived agents accumulating operational experience that cannot be represented adequately as an unordered document collection.

Suggested contribution framing:

1. a heterogeneous execution-memory topology separating semantic, causal, temporal and behavioral relations;
2. provenance-preserving causal paths derived from runtime/event evidence;
3. basin/attractor organization of historical trajectories;
4. a non-authoritative terrain whose retrieval influence can reinforce, erode and drift over time;
5. staged retrieval that composes vector/lexical anchoring with structural navigation;
6. controlled experiments for root-cause localization, recovery retrieval and early behavioral degradation.

## Counterfactual retrieval versus causal counterfactual identification

CT-RAG supports **counterfactual retrieval** before it supports **counterfactual causal identification**.

Consider the currently dominant observed basin:

```text
PaymentFailure
  -> Retry
  -> Timeout
  -> HumanIntervention
```

The historical terrain may contain an alternative trajectory:

```text
PaymentFailure
  -> Retry
  -> ProviderFallback
  -> Recovered
```

even if that recovery path currently has low navigational influence because it has not been used recently.

A COUNTERFACTUAL retrieval query can therefore ask:

> Under historically similar conditions, which alternative observed trajectories diverged from the current failure basin and reached a recovery attractor?

The retrieval procedure can:

1. anchor the current failure state;
2. identify semantically/structurally comparable historical states;
3. locate divergence points;
4. traverse alternative historical branches;
5. rank paths reaching a desired attractor such as `Recovered`;
6. expose the conditions, provenance and evidence attached to those paths.

This is useful for hypothesis generation and recovery planning.

However, the result is still observational evidence. It does not establish:

> If the system had taken ProviderFallback now, it would have recovered.

That stronger statement requires an interventional or otherwise identified causal model.

## Eroded paths remain retrievable

Dynamic terrain affects retrieval strength, not historical existence.

Therefore a path can be:

```text
historically preserved
+
currently low influence
+
still queryable when explicitly relevant
```

This is essential for counterfactual retrieval. Rare or eroded successful paths should not disappear merely because current behavior has converged toward a degraded basin.

For critical paths, the existing protected-edge floor prevents navigational decay below a configured minimum. For ordinary historical paths, explicit counterfactual/recovery mode can relax current-terrain priors and prioritize structural equivalence, causal provenance and target-attractor reachability.

## Research hypothesis for counterfactual retrieval

A concrete falsifiable hypothesis is:

> Given a current degraded basin, CT-RAG retrieves historically observed alternative trajectories that reached a desired attractor with higher causal-path relevance and lower irrelevant-context rate than semantic retrieval alone.

A stronger future hypothesis, once intervention-aware data exists, is:

> Intervention-labeled historical trajectories improve the precision of alternative-path retrieval relative to observational trajectories alone.

## Suggested paper section title

**Counterfactual Retrieval over Experiential Topology**

Recommended subsections:

- Observed branch alternatives
- Divergence-point retrieval
- Eroded-path recovery
- Intervention-aware evidence
- Boundary between retrieval and causal identification


## Scope recommendation: one system, two papers

The repository can contain all mechanisms, but the first submission should not attempt to prove every research direction simultaneously.

### Paper 1 — CT-RAG core

Primary contribution:

- Structured Experiential Memory for long-lived agents;
- runtime/event causal provenance;
- causal/temporal/behavioral separation;
- staged navigation;
- basins and attractors;
- non-authoritative adaptive terrain;
- root-cause, recovery and behavioral-drift retrieval.

Primary evaluation:

- controlled ablations;
- real external execution/telemetry validation;
- sensitivity/null/bootstrap robustness;
- retrieval and trajectory metrics.

### Paper 2 — Intervention-aware longitudinal causal memory

Primary contribution:

- intervention records for deploy/config/feature-flag/rollback events;
- pre/post basin-distribution analysis;
- quasi-experimental identification;
- intervention-conditioned retrieval;
- inferred causal discovery with explicit weaker provenance;
- stronger counterfactual evaluation.

This second line may ultimately be the stronger causal contribution, but including it as a full first-paper claim would require a second independent validation burden.

### Evidence Shape Router

Structure-preserving evidence retrieval should be treated as an orthogonal retrieval mechanism until the factorial ablation establishes whether it adds independent value to CT-RAG.

The first paper can describe it as an extensibility layer or secondary experiment. It should become a headline contribution only if the 2x2 ablation demonstrates an identifiable main effect or useful interaction.
