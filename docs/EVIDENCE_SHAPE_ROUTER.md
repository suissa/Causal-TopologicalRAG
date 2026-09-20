# Evidence Shape Router

## Purpose

The Evidence Shape Router decides **how evidence should be read and retrieved before CT-RAG performs experiential navigation**.

Its central rule is:

> Do not destroy artifact-native structure merely to force every evidence item through the same embedding pipeline.

CT-RAG primarily models relations **between experiences**. The Evidence Shape Router preserves useful structure **inside each evidence artifact**.

```text
Evidence artifact
      |
      v
Evidence Shape Router
      |
      v
shape-specific retrieval adapter
      |
      v
Typed Evidence Anchor
      |
      v
CT-RAG experiential topology
```

## Two topologies

### Intrinsic topology

Structure native to one artifact.

Examples:

```text
table    -> row / column / cell / header
config   -> tree / path / key / value
trace    -> trace / span / parent-child / duration
metric   -> timestamp / series / window / change point
code     -> AST / symbol / call / dependency
image    -> region / object / spatial relation
```

### Experiential topology

Relations between memories, states and executions.

```text
event
  -> causal
  -> temporal
  -> behavioral
  -> trajectory
  -> basin
  -> attractor
```

Intrinsic structural adjacency never automatically becomes causal evidence.

```text
artifact adjacency
    !=
temporal adjacency
    !=
behavioral continuity
    !=
causal evidence
```

## EvidenceShape

Reference taxonomy:

```text
NARRATIVE
TABLE
CONFIG_TREE
TRACE
METRIC_SERIES
EVENT
LOG
CODE
GRAPH
IMAGE
STRUCTURED_PAYLOAD
UNKNOWN
```

The taxonomy is extensible. Shape is a routing property, not an epistemic claim.

## Routing policy

The router should prefer deterministic classification whenever source metadata or schema is sufficient.

Examples:

```text
OpenTelemetry span       -> TRACE
Prometheus time series   -> METRIC_SERIES
YAML/TOML config         -> CONFIG_TREE
EventRecord              -> EVENT
relational table         -> TABLE
source file + AST        -> CODE
```

Only ambiguous inputs should require semantic/LLM classification.

Recommended decision path:

```text
schema / MIME / source metadata
          |
          v
deterministic classifier
          |
      confidence high?
       /          \
     yes           no
      |             |
      v             v
    route     semantic classifier
                       |
                       v
                     route
```

## Router contract

A route decision should contain at least:

```json
{
  "shape": "TRACE",
  "confidence": 0.98,
  "classifier": "deterministic:opentelemetry",
  "adapter": "trace-span",
  "preserve": [
    "trace_id",
    "span_id",
    "parent_span_id",
    "service",
    "operation",
    "start_time",
    "duration",
    "status"
  ]
}
```

The router must never silently discard fields required for provenance, chronology or causal validation.

## TypedEvidenceAnchor

Shape-specific adapters should emit a normalized anchor rather than flattening the source into untyped prose.

Conceptual contract:

```text
TypedEvidenceAnchor
  id
  shape
  source_id
  source_uri
  observed_at
  content_summary
  structural_locator
  preserved_fields
  local_relations
  retrieval_features
  provenance
  confidence
```

`structural_locator` is shape-specific:

```text
CONFIG_TREE   -> inventory.reservation.timeout_ms
TRACE         -> trace_id/span_id
TABLE         -> table/row/column/cell
METRIC_SERIES -> series/window/timestamp range
CODE          -> file/symbol/AST node
IMAGE         -> region/object coordinates
```

## Shape-specific adapters

### NARRATIVE

Use dense, lexical or hybrid retrieval. Preserve document/section identity when available.

### TABLE

Preserve cell, row, column and header relations. Multi-vector or late-interaction retrieval is preferred when whole-table compression is demonstrably lossy.

### CONFIG_TREE

Preserve hierarchical paths, value types, inherited values, source layer and effective-value resolution.

Example:

```text
inventory.reservation.timeout_ms = 3000
```

should not be reduced to an anonymous token sequence when the path itself is diagnostic evidence.

### TRACE

Preserve trace/span identity, parent-child hierarchy, service, operation, duration and status.

Important invariant:

> span parent-child is structural execution evidence; it is not automatically an `E_c` causal edge.

### METRIC_SERIES

Preserve the time axis and windows. Retrieval features can include trend, slope, anomaly score, quantiles, change points and seasonality.

### EVENT

Preserve event identity, execution/correlation/causation IDs and authoritative event time. Explicit runtime `causation_id` can be promoted to causal topology according to CT-RAG's provenance policy.

### LOG

Preserve timestamp, logger/service, severity, structured fields, correlation IDs and message template. Free text remains available for lexical/semantic retrieval.

### CODE

Preserve AST/symbol/module/dependency structure and source positions. Similarity between code fragments must not be interpreted as execution causation.

### GRAPH

Preserve native node/edge types and provenance. Imported graph edges retain their source semantics instead of being coerced into CT-RAG causal edges.

### IMAGE

Preserve regions, objects, captions and spatial relationships when the upstream vision adapter provides them.

## Evidence requirements before routing

The router should not indiscriminately retrieve every shape. An Intent or diagnostic policy can state which evidence shapes are relevant.

```text
Intent: DiagnoseCheckoutFailure

EvidenceRequirements:
  - EVENT
  - TRACE
  - METRIC_SERIES
  - LOG
  - CONFIG_TREE
```

Then:

```text
Intent
  -> Evidence Requirements
  -> Evidence Shape Router
  -> specialized adapters
  -> Typed Evidence Anchors
  -> CT-RAG
```

## Example: cross-artifact diagnosis

Input evidence:

```text
Config:
  inventory.reservation.timeout_ms = 3000

Metric:
  inventory_latency_p99 = 4600ms

Trace:
  Inventory.reserve = 4800ms

Log:
  reservation timeout

Event:
  Inventory.ReservationFailed

Error:
  Checkout failed
```

The router preserves each source according to its shape:

```text
CONFIG_TREE
  -> path/value/type

METRIC_SERIES
  -> window/trend/anomaly

TRACE
  -> span hierarchy/duration

LOG
  -> structured fields/template

EVENT
  -> event/execution/causation identity
```

CT-RAG can then connect the resulting anchors through experiential topology without erasing the distinction between the original sources.

## Integration with CodeManager

Recommended end-to-end flow:

```text
Natural-language request
        |
        v
Intent classification
        |
        v
Evidence Requirements
        |
        v
Evidence Shape Router
        |
        v
shape-specific retrieval adapters
        |
        v
Typed Evidence Anchors
        |
        v
CT-RAG
  causal / temporal / behavioral navigation
  basin / attractor / terrain context
        |
        v
CodeManager
  evidence correlation
  hypothesis generation
  confidence + provenance
```

The router answers:

> How should this evidence be read?

CT-RAG answers:

> How is this evidence related to the system's experience?

CodeManager answers:

> Which hypothesis best explains the evidence jointly?

## AllasCode/A3 implementation mapping

A compatible action decomposition is:

```text
EvidenceShapeAgent.Detect
EvidenceShapeAgent.Route

ConfigEvidence.Retrieve
TraceEvidence.Retrieve
MetricEvidence.Retrieve
LogEvidence.Retrieve
EventEvidence.Retrieve
CodeEvidence.Retrieve
```

Each Action can own an atomic skill describing accepted source shape, preserved invariants, normalization rules and failure/healing behavior.

## Invariants

1. **Structure preservation:** do not flatten away retrieval-relevant native structure without an explicit baseline/ablation reason.
2. **No causal promotion by shape:** local adjacency never creates causal authority.
3. **Provenance preservation:** source identity and transformation lineage survive routing.
4. **Deterministic-first:** use schemas/metadata before an LLM classifier when possible.
5. **Ambiguity is explicit:** low-confidence routes may return multiple candidate adapters or `UNKNOWN` rather than inventing certainty.
6. **Adapter isolation:** a shape adapter produces evidence anchors; it does not mutate authoritative CT-RAG history.
7. **Typed output:** downstream systems should know whether an anchor came from a metric, trace, config, event, log, table, code or another shape.
8. **Reproducibility:** classification version, adapter version and transformation parameters must be persisted in research artifacts.

## Evaluation

The router should be evaluated independently from CT-RAG ranking.

Suggested metrics:

```text
shape classification accuracy
routing accuracy
field-preservation recall
structural-locator accuracy
adapter retrieval Recall@K
end-to-end root-cause localization
context compression ratio
causal-contamination errors
```

Critical negative test:

> An artifact-local relation must never appear as an observed CT-RAG causal edge unless an independent causal provenance source authorizes that promotion.

## Relation to Topo-RAG

Topo-RAG motivates preserving artifact-native structure, particularly text/table topology. CT-RAG generalizes the architectural lesson across operational evidence shapes, while keeping intrinsic artifact topology separate from experiential causal-topological memory.

## Status

This document is the architectural contract for the proposed router. Shape-specific retrieval exists today in separate forms/adapters across the project, but a single production `EvidenceShapeRouter` module implementing the complete contract remains future implementation work.