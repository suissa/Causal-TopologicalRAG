# Optional Cognitive Graph

## Status

The Cognitive Graph is an **optional** memory/database layer. It is not required for CT-RAG core operation and it is intentionally separate from CTEG.

> **CTEG stores what happened. The Cognitive Graph stores what the agent believed, intended, hypothesized, predicted, and decided about what happened.**

## Why a separate database

Observed experience and cognitive interpretation have different epistemic status.

Example:

```text
CTEG
────────────────────────
Metric.cpu = 98%
Log = worker timeout
Event = Payment.Timeout
Trace = checkout -> payment -> worker

Cognitive Graph
────────────────────────
Hypothesis = database saturation
confidence = 0.72
Decision = restart DB connection pool
Goal = restore payment processing
```

If later evidence shows worker memory pressure was the actual causal source, the hypothesis must remain a historical cognitive state rather than contaminating CTEG as authoritative causal truth.

## Candidate cognitive entities

```text
Belief
Goal
Hypothesis
Decision
Plan
Prediction
Uncertainty
Reflection
Attention
```

These are optional. A deployment that needs only experiential memory should not have to implement them.

## Relationship to CTEG

The Cognitive Graph may reference CTEG evidence without changing its authority:

```text
Hypothesis H42
    ├── based_on -> Trace T51
    ├── based_on -> Metric M18
    ├── based_on -> Log L09
    └── predicts -> Event E73
```

A later observation may update confidence in H42, but it must not rewrite the underlying CTEG evidence.

## Optional cognitive terrain

A future implementation may maintain a distinct Cognitive Terrain:

```text
Experiential Terrain
→ which observed paths are currently influential

Cognitive Terrain
→ which hypotheses, plans or strategies are gaining or losing confidence
```

These terrains must remain separate because navigational frequency and epistemic confidence are not the same quantity.

## Architectural rule

```text
CTEG Database       REQUIRED
Cognitive Graph     OPTIONAL
```

The core CT-RAG package must never require cognitive entities to use WHY, RECOVERY, WHAT_NEXT, SIMILAR, or experiential terrain navigation.
