# Intervention-Aware CT-RAG

## Motivation

Software systems naturally emit exogenous change events:

- deploy;
- configuration change;
- feature-flag change;
- rollback;
- dependency version change;
- traffic-routing policy change;
- operator intervention.

These events can create useful before/after boundaries for causal analysis.

## Intervention record contract

A future intervention record should minimally contain:

```text
intervention_id
kind
target_scope
effective_time
actor/source
old_value/reference
new_value/reference
correlation_id
deployment/config version
```

The record is evidence that an intervention occurred. It is **not** evidence that the intervention caused a later outcome.

## Identification contract

Any causal-effect estimate associated with an intervention must also record:

```text
estimand
treated population/scope
control or comparison
pre window
post window
assumptions
diagnostics
effect estimate
uncertainty
method
evidence ids
```

For difference-in-differences, for example, the parallel-trends assumption must be stated and checked where possible.

## Basin-level analysis

Let `P_pre` and `P_post` be transition models estimated on windows before and after an intervention. Let `pi_pre(A|x)` and `pi_post(A|x)` be absorption distributions over attractors.

An intervention-associated terrain change can be summarized by TV/JS divergence between these distributions. This is still descriptive unless the identification design supports an intervention-level causal claim.

## Why this matters

This turns deploy/config/flag history into a bridge between:

```text
observational drift
        ↓
quasi-experimental identification
        ↓
counterfactual hypotheses
```

without collapsing those levels into one another.