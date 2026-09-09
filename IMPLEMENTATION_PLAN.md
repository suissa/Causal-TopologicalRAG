# Implementation Plan

## Goal

Build a reproducible Causal-Topological RAG (CT-RAG) research framework that can retrieve not only semantically similar memories, but also causally and topologically relevant execution history for stateful agents and event-driven systems.

## Non-goals for v0.1

- claiming causal relationships from timestamp order alone;
- replacing dedicated vector databases or graph databases;
- requiring an LLM during indexing;
- solving general causal inference or counterfactual identification;
- production-scale distributed storage.

## Core invariants

1. **Temporal adjacency is not causality.**
2. **Every causal edge has provenance and confidence.**
3. **Observed and inferred causal relations remain distinguishable.**
4. **The authoritative event history is not replaced by the retrieval projection.**
5. **Retrieval can explain its score components.**
6. **Query intent changes traversal direction and scoring weights.**
7. **Basins/attractors are queryable structures, not decorative visualization metadata.**

## Phase 0 — Executable research skeleton

Deliverables:

- Python package layout and CI;
- typed memory nodes, edges, provenance and query modes;
- deterministic local embedding baseline;
- lexical baseline;
- causal/topological graph traversal;
- explicit attractors and basin computation;
- hybrid CT-RAG scoring;
- Event Sourcing projector;
- minimal example and unit tests.

Acceptance criteria:

- `pytest` passes on Python 3.11+;
- a `WHY` query can prefer an explicit causal ancestor over an unrelated semantically similar memory;
- reverse traversal can recover the basin of an attractor;
- temporal edges never become causal implicitly.

## Phase 1 — Retrieval quality baselines

Implement interchangeable retrieval adapters and establish baselines:

- BM25;
- dense embeddings (Sentence Transformers/OpenAI-compatible adapter, optional);
- reciprocal-rank fusion;
- vector-only baseline;
- graph-only baseline;
- CT-RAG staged retrieval;
- score calibration and ablation toggles.

Benchmark the contribution of each signal:

```text
Dense
Dense + Lexical
Dense + Causal
Dense + Topological
Dense + Causal + Topological
Full CT-RAG
```

## Phase 2 — Causal provenance model

Extend causal evidence beyond a scalar confidence:

- evidence IDs and source references;
- provenance classes (`execution`, `workflow`, `dependency`, `event`, `inferred`, `hypothesized`);
- confidence calibration by provenance;
- path confidence aggregation;
- causal-edge invalidation/versioning;
- graph audit trail.

Research question: does provenance-aware retrieval reduce unsupported causal explanations from the downstream LLM?

## Phase 3 — Dynamic terrain and basin discovery

Move from manually registered attractors to empirical terrain analysis:

- sink/recurrent-state discovery;
- strongly connected components;
- transition-frequency reinforcement;
- edge decay/erosion policies;
- basin stability over time;
- attractor confidence;
- execution-trajectory clustering without collapsing causality into semantic similarity.

Potential formulations:

- deterministic functional graphs;
- Markov transition fields;
- weighted directed graphs;
- energy/landscape approximations.

## Phase 4 — Event Store adapters

Create ingestion/projector adapters for real event-sourced systems:

- generic NDJSON event log;
- EventStoreDB;
- SQLite/local event log baseline;
- user-provided event schema mapping.

Required fields should support, when available:

```text
event_id
causation_id
correlation_id
execution_id
intent_id
actor_id
action_id
timestamp
status
```

The retrieval graph remains a projection. The event store remains authoritative.

## Phase 5 — Stateful-agent retrieval

Add agent-oriented operations:

- reconstruct current execution trajectory;
- retrieve causal ancestors of a failure;
- retrieve historical recovery paths;
- find similar states inside the same/neighboring basin;
- compare successful vs failed trajectories;
- find divergence points;
- retrieve likely next states from observed transitions.

Proposed API:

```python
retriever.why(state_id)
retriever.what_next(state_id)
retriever.recovery_paths(state_id)
retriever.similar_trajectories(execution_id)
retriever.divergence(success_execution, failed_execution)
```

## Phase 6 — Benchmark suite

Issue #7 implemented: `python -m ctrag.benchmarks` executes seven controlled
baseline/ablation arms on deterministic failure/recovery and branching traces,
including WHY, RECOVERY and WHAT_NEXT queries. It persists seeds, effective
weights, full datasets/labels, source fingerprints and JSON/CSV observations plus
paper-table summaries. CI executes the full suite and uploads artifacts.

The harness uses exhaustive candidates and known anchors for fair score ablation.
Its dense arm is the existing hashing proxy, not a learned embedding benchmark.
See [methodology and exact metric definitions](docs/benchmarks.md).

Remaining research work: real trace fixtures, learned dense/BM25 adapters,
anchor-discovery evaluation, prospective tasks without future-state visibility,
larger/more varied graphs and statistical hypothesis testing. Synthetic harness
completion alone does not validate the core hypothesis below.

Primary metrics:

- Recall@K / Precision@K / MRR / nDCG;
- Causal Recall@K;
- Causal Path Recall;
- Causal Distance Error;
- Trajectory Reconstruction Accuracy;
- Basin Purity;
- Recovery Path Precision;
- context-token efficiency.

Core hypothesis:

> For diagnostic and recovery queries over stateful execution histories, causal-topological retrieval recovers more causally relevant context with fewer irrelevant memories than semantic nearest-neighbor retrieval alone.

## Phase 7 — Persistence and scale

Introduce storage interfaces before choosing infrastructure:

- `MemoryStore`;
- `VectorIndex`;
- `TopologyStore`;
- `EventSource`;
- `TerrainStore`.

Then benchmark implementations such as:

- Qdrant/FAISS for vectors;
- Neo4j/CozoDB or adjacency-store implementations for topology;
- EventStoreDB for authoritative event history;
- SQLite/DuckDB for local experiments.

Do not couple the semantic model to a specific database.

## Phase 8 — Research artifacts

- formal problem statement;
- mathematical notation for CT-RAG;
- reproducible benchmark scripts;
- ablation tables;
- visualization of terrain, basins and causal paths;
- comparison with Vector RAG, GraphRAG and BasinRAG;
- paper-ready figures and experiment manifests.

## Proposed issue breakdown

1. Core model and invariants
2. Hybrid semantic/lexical baseline
3. Causal topology and path-confidence traversal
4. Basin/attractor engine
5. Event Sourcing projector and NDJSON ingestion
6. Query modes and staged CT-RAG retriever
7. Benchmark and ablation harness
8. Dynamic terrain reinforcement/erosion
9. Persistent storage interfaces and adapters
10. Research documentation, visualization and reproducibility

## v0.1 definition of done

CT-RAG v0.1 is complete when a fresh clone can install the package, run the example and tests, ingest an event trace with explicit causation, identify a current state as an anchor, traverse its causal/topological neighborhood, and produce an explainable ranked context showing why each memory was retrieved.
