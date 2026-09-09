# Causal-Topological RAG — Implementation Plan

## Objective

Deliver CT-RAG as a reproducible research implementation for retrieving execution memories by semantic, lexical, causal, temporal, behavioral and topological evidence without collapsing temporal adjacency into causality.

The implementation is considered research-valid only when every retrieval signal is independently testable, every causal claim preserves provenance, the benchmark is reproducible, and reported results can be regenerated from committed code/configuration.

## v0.1 status

**Implementation roadmap #1–#10: complete.**

The repository now has a tested path from authoritative execution events to causal/topological retrieval, empirical terrain evolution, persistence and reproducible paper artifacts.

The controlled experiment in [`REPORT.md`](REPORT.md) provides positive proof-of-concept evidence on synthetic event-sourced traces. External validity remains a separate post-v0.1 research phase.

## Core invariants

1. Temporal adjacency MUST NOT imply causality.
2. Every causal edge MUST carry provenance and confidence.
3. Observed, inferred and hypothesized causality MUST remain distinguishable.
4. Authoritative event history MUST remain separate from retrieval projections.
5. Retrieval results MUST expose score components and structural evidence.
6. Query intent MUST control traversal direction and weighting.
7. Basins and attractors MUST be queryable/testable structures rather than visualization-only metadata.
8. Reinforcement/erosion MUST change navigational influence without silently rewriting history.
9. Database adapters MUST preserve CT-RAG semantics.
10. Benchmark claims MUST state their scope and limitations.

## Execution policy used

Every tracked implementation issue was completed using this sequence:

1. inspect current implementation against acceptance criteria;
2. implement the smallest complete semantic slice;
3. add/strengthen tests;
4. update public documentation/contracts;
5. commit the issue work;
6. verify GitHub Actions on Python 3.11–3.13;
7. add issue completion evidence;
8. close only after criteria passed.

## Completed issue roadmap

### #1 — Core memory model and invariants — COMPLETE

Delivered:

- strict node/edge identity validation;
- edge evidence/provenance metadata;
- deterministic serialization/deserialization;
- confidence/weight/provenance validation;
- duplicate-edge protection;
- public model contracts and tests.

### #2 — Hybrid semantic + lexical adapters — COMPLETE

Delivered:

- `EmbeddingProvider` and `LexicalRetriever` contracts;
- deterministic `HashingEmbedder` fallback;
- IDF overlap and BM25;
- optional Sentence Transformers adapter;
- OpenAI-compatible embedding adapter;
- deterministic Reciprocal Rank Fusion;
- independent dense, lexical and hybrid ranking APIs.

### #3 — Causal path confidence and provenance weighting — COMPLETE

Delivered:

- provenance-calibrated causal confidence;
- best path reconstruction plus multipath aggregate confidence;
- ancestor/descendant traversal budgets;
- cycle-safe causal traversal;
- `CausalPath` evidence on retrieval hits;
- strict exclusion of temporal/behavioral edges from causal confidence.

### #4 — Basin and attractor engine — COMPLETE

Delivered:

- `AttractorDescriptor` with confidence/origin/metadata;
- basin membership;
- shared-basin affinity;
- basin boundaries;
- neighboring basins;
- branching, convergence, disconnected-component and cycle tests.

### #5 — Event Sourcing and NDJSON contracts — COMPLETE

Delivered:

- validated `EventRecord`;
- configurable/nested `EventFieldMapping`;
- canonical event fingerprints;
- deterministic idempotent replay;
- explicit conflict detection for reused event IDs;
- pending/out-of-order `causation_id` reconciliation;
- explicit causal edge evidence;
- temporal/behavioral sequence without implicit causality;
- NDJSON error line/event identity;
- multi-step failure/healing integration fixture.

### #6 — Staged query modes — COMPLETE

Delivered staged retrieval:

```text
anchor search
  -> basin/topology expansion
  -> directed causal traversal
  -> mode-specific reranking
```

Modes:

- `WHY`: causal ancestors;
- `WHAT_NEXT`: causal descendants;
- `RECOVERY`: observed successful recovery paths;
- `COUNTERFACTUAL`: historical divergence with explicit observational-only disclaimer.

`RetrievalStage` and `StagedRetrievalResult` expose anchors, stages, component scores and path evidence.

### #7 — Benchmark and ablation harness — COMPLETE

One command:

```bash
python -m ctrag.benchmarks
```

reproduces seven controlled arms:

```text
lexical_only
dense_only
dense_lexical
graph_topology_only
dense_causal
dense_topological
full_ctrag
```

Metrics include:

- Recall@K / Precision@K / MRR / nDCG;
- Causal Recall@K;
- Causal Path Recall;
- Causal Distance Error;
- Trajectory Reconstruction Accuracy;
- Basin Purity;
- Recovery Path Precision;
- context-token efficiency.

Outputs persist seeds, effective weights, generated datasets/labels, config, source fingerprints and JSON/CSV/paper-table data. [`REPORT.md`](REPORT.md) records the controlled validation and its limitations.

### #8 — Dynamic terrain — COMPLETE

Delivered:

- transition-frequency tracking;
- non-authoritative navigational influence;
- reinforcement policies;
- exponential erosion/decay;
- deterministic Tarjan strongly connected components;
- sink/recurrent attractor discovery;
- explicit `discovered:sink` / `discovered:scc` origins;
- protection of manually declared attractors;
- basin snapshots and Jaccard drift;
- `TerrainAwareRetriever` final reranking without rewriting authoritative edge semantics.

### #9 — Persistence and storage interfaces — COMPLETE

Delivered contracts:

```text
TopologyView
MemoryStore
VectorIndex
TopologyStore
EventSource
TerrainStore
```

Reference adapters:

- `InMemoryVectorIndex`;
- `ListEventSource`;
- `SQLiteCTStore`.

SQLite persists/reloads nodes, timestamps, metadata, embeddings, all edge kinds, causal provenance/confidence/weight/evidence, attractor descriptors and terrain overlay/config.

Conformance tests prove semantic serialization equality and identical CT-RAG ranking/score components/causal hops before and after reload.

See [`docs/STORAGE.md`](docs/STORAGE.md).

### #10 — Research visualization and reproducibility — COMPLETE

Delivered:

- [`docs/RESEARCH.md`](docs/RESEARCH.md): formal problem statement, notation, epistemic causal distinctions and sourced related work;
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): exact end-to-end protocol;
- `ctrag.research_artifacts`: deterministic paper-artifact generator;
- topology DOT figure source;
- ground-truth causal-path DOT figure source;
- benchmark Markdown table generated from `table.csv`;
- manifest with exact commands/config plus SHA-256 input/output fingerprints;
- CI generation/upload of research artifacts for Python 3.11–3.13;
- README research-artifact index.

Generated artifact command:

```bash
python -m ctrag.research_artifacts \
  --benchmark-dir benchmark-results \
  --out research-artifacts \
  --k 3
```

## v0.1 definition of done

A fresh clone can execute:

```bash
pip install -e ".[dev]"
pytest
python -m ctrag.benchmarks
python -m ctrag.research_artifacts --benchmark-dir benchmark-results --out research-artifacts --k 3
```

and can additionally:

- ingest/replay event traces idempotently;
- reconcile explicit causation arriving out of order;
- preserve observed/inferred/hypothesized provenance;
- reconstruct explainable causal paths;
- retrieve WHY/WHAT_NEXT/RECOVERY/COUNTERFACTUAL context;
- inspect and discover basins/attractors;
- reinforce/erode navigational terrain without rewriting history;
- persist/reload semantic retrieval state;
- reproduce benchmark tables and research visualizations from machine outputs.

## Research validation beyond v0.1

The controlled synthetic result in `REPORT.md` demonstrates the architecture on authored event-sourced traces. It does not establish universal superiority.

Post-v0.1 external validation should add:

- real event-sourced traces;
- strong production BM25 baseline;
- learned dense embedding baselines;
- GraphRAG/BasinRAG comparisons under matched datasets and context budgets;
- anchor-discovery evaluation rather than known anchors only;
- prospective tasks without future-state visibility;
- larger graphs and independent datasets;
- confidence intervals and preregistered statistical comparisons;
- interventional datasets before making counterfactual-causality claims.

## Phase 2 — Scientific validation and falsification — ACTIVE

The post-v0.1 research phase is now tracked by issue **#35** and by [`SCIENTIFIC_VALIDATION_PLAN.md`](SCIENTIFIC_VALIDATION_PLAN.md).

The active scientific backlog is **#14–#34**, grouped into six gates:

1. **Internal validity/freeze:** #14, #15, #18, #33.
2. **Competitive baselines/anchor realism:** #16, #17, #21, #22.
3. **External validity:** #19, #20, #31.
4. **Causal/topological falsification:** #23, #24, #27, #30.
5. **Dynamics/downstream/systems evidence:** #26, #28, #29.
6. **Inference/replication/final claims:** #25, #32, #34.

### Phase 2 scientific policy

- A negative result is not automatically a bug.
- Any correctness bug exposed by an experiment must first receive a minimal failing regression test.
- Logic corrections are recorded in `CHANGELOG_SCIENCE.md` and evaluated on train/dev before any new final-holdout run.
- Headline claims require strong baselines, uncertainty/effect size, and evidence beyond the authored synthetic generator.
- Destruction/placebo controls must test whether gains actually depend on causal/topological information.
- Prospective tracks must not use future-state visibility.
- Observational divergence must not be described as identified counterfactual effect.
- Final scientific claims are bounded by the claim matrix in `SCIENTIFIC_VALIDATION_REPORT.md`.

### Phase 2 definition of done

Issue #35 may be closed only when the preregistered primary hypotheses are evaluated on frozen held-out data, strong compatible baselines have been run, independent/real data contributes to the evidence, topology falsification controls are complete, statistical uncertainty is reported, clean-room replication succeeds, and #34 produces the final claim-bounded scientific validation report.
