# Related Work Note: Topo-RAG (arXiv:2601.10215)

## Why this paper matters to CT-RAG

The paper **"Topo-RAG: Topology-aware retrieval for hybrid text-table documents"** (arXiv:2601.10215) uses the word *topology* in a materially different sense from CT-RAG.

Topo-RAG focuses on the **intrinsic structure of the retrieved artifact**:

- narrative text follows a dense-retrieval path;
- tables follow a cell-aware multi-vector / late-interaction path;
- row/column/cell position is preserved instead of flattening the table into one text string;
- a topology-aware router selects the appropriate retrieval path;
- candidates are unified by a reranker.

CT-RAG focuses on the **experiential structure of system memory**:

- causal relationships between events/states;
- temporal order;
- behavioral/execution relationships;
- trajectories;
- basins and attractors;
- provenance;
- dynamic terrain and basin drift.

These two meanings of topology are complementary.

## Two distinct topologies

We therefore distinguish:

### 1. Intrinsic data topology

Structure that belongs to the artifact itself.

Examples:

```text
table:
  row
  column
  cell
  header
  coordinate

document:
  section
  paragraph
  list
  figure
  caption
```

Topo-RAG is primarily in this category.

### 2. Experiential topology

Structure that belongs to the system's history and observed execution.

Examples:

```text
event
  -> causal successor
  -> temporal successor
  -> behavioral successor
  -> recovery trajectory
  -> attractor
```

CT-RAG is primarily in this category.

A future production system can preserve both simultaneously.

## Composite memory model

A useful extension for CT-RAG is to separate the topology of the memory item from the topology between memory items.

Let:

[
G_X=(V_X,E_X)
]

be the experiential topology, and let each memory artifact (m) optionally contain an intrinsic structure:

[
T(m)=(U_m,R_m)
]

where (U_m) can represent cells, fields, sections or other structural units and (R_m) their native relations.

Retrieval can then be staged:

```text
query
  -> modality / structure routing
  -> artifact-local retrieval
  -> memory anchor
  -> experiential topology expansion
  -> causal / temporal / behavioral traversal
  -> reranking
```

This prevents CT-RAG from making the same mistake it criticizes elsewhere: destroying useful structure before retrieval.

## Direct lessons from Topo-RAG

### Preserve structure before embedding

Topo-RAG argues against flattening rich tabular structures into a single linear representation. The corresponding CT-RAG rule should be:

> Do not destroy artifact-native structure merely to make every memory fit the same embedding pipeline.

For structured evidence such as metrics, traces, logs, configuration trees, tables and spans, the adapter should preserve relevant native coordinates/relations.

### Route before retrieval

Topo-RAG uses topology-aware routing before the retrieval stage. CT-RAG can generalize this idea into an **evidence-shape router**:

```text
natural language -> dense / lexical
table            -> cell-aware / late interaction
trace             -> span / parent-child structure
metric            -> series / window / anomaly structure
config            -> tree / path structure
event             -> causal-temporal graph
```

The router must not change causal authority. It only chooses the retrieval representation appropriate to the evidence shape.

### Use late interaction when compression is lossy

Single-vector compression can erase fine-grained relations. CT-RAG should permit multi-vector or late-interaction adapters for structured evidence where one-vector-per-artifact is demonstrably lossy.

This is particularly relevant to:

- configuration trees;
- structured error payloads;
- tabular telemetry;
- span-rich traces;
- metric windows.

### Keep retrieval fusion explicit

Topo-RAG converges specialized retrieval routes through a unified reranker. CT-RAG should retain the same architectural discipline:

```text
specialized retrievers
        |
        v
typed candidate evidence
        |
        v
CT-RAG staged topology expansion
        |
        v
explainable reranking
```

## Important non-equivalence

Topo-RAG does **not** provide causal topology.

Cell adjacency or row/column structure must never be treated as evidence that one observation caused another.

Likewise, CT-RAG's event topology does not automatically preserve the internal spatial structure of a table.

Therefore:

```text
intrinsic structural adjacency
    !=
temporal adjacency
    !=
behavioral continuity
    !=
causal evidence
```

## Research opportunity

A combined benchmark would be valuable:

1. construct incidents whose evidence is distributed across narrative text, a table/config tree and an execution trace;
2. compare flatten-all RAG against structure-aware local retrieval;
3. then compare local retrieval alone against CT-RAG causal/topological expansion;
4. measure both answer retrieval quality and root-cause localization.

This would test whether preserving **artifact topology** and **experiential topology** provides complementary gains.

## Citation

Alex Dantart and Marco Kóvacs-Navarro. *Topo-RAG: Topology-aware retrieval for hybrid text-table documents*. arXiv:2601.10215v1, 2026.

https://arxiv.org/abs/2601.10215
