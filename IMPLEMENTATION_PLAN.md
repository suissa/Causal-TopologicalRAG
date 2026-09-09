# Causal-Topological RAG — Implementation Plan

## Objective

Deliver CT-RAG as a reproducible research implementation for retrieving execution memories by semantic, lexical, causal, temporal, behavioral and topological evidence without collapsing temporal adjacency into causality.

The implementation is considered research-valid only when every retrieval signal is independently testable, every causal claim preserves provenance, the benchmark is reproducible, and reported results can be regenerated from committed code/configuration.

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

## Current baseline

Already implemented before this execution pass:

- typed memory nodes and edge kinds;
- deterministic HashingEmbedder and lexical overlap baseline;
- causal/topological graph traversal;
- explicit attractor registration and basin traversal;
- hybrid CT-RAG scoring;
- Event Sourcing projection with explicit `causation_id`;
- synthetic benchmark harness with seven ablation arms;
- deterministic failure/recovery and branching datasets;
- JSON/CSV reproducibility artifacts;
- CI on Python 3.11, 3.12 and 3.13;
- `REPORT.md` with the first controlled validation results.

The benchmark issue (#7) is functionally implemented but remains open until it is explicitly reconciled against its acceptance criteria during this pass.

## Execution policy

Issues are resolved in dependency order. Each issue is completed using this sequence:

1. inspect current implementation against the issue acceptance criteria;
2. implement the smallest complete semantic slice;
3. add or strengthen tests;
4. update public documentation/contracts when behavior changes;
5. commit the issue independently;
6. verify GitHub Actions;
7. add an issue completion note with evidence;
8. close the issue only after its criteria are satisfied.

A later issue may reuse APIs introduced by earlier issues, but earlier issue semantics must not be weakened.

## Issue-by-issue roadmap

### #1 — Harden core memory model and CT-RAG invariants

Deliverables:

- strict node and edge identity validation;
- edge evidence/provenance metadata;
- serialization/deserialization contracts;
- validation for confidence/weight and invalid provenance combinations;
- public model API documentation;
- round-trip and invariant tests.

Exit condition: model contracts are deterministic and invalid causal representations fail fast.

### #2 — Hybrid semantic + lexical retrieval adapters

Deliverables:

- `EmbeddingProvider` protocol;
- `LexicalRetriever` protocol;
- deterministic HashingEmbedder fallback;
- BM25 implementation;
- optional Sentence Transformers provider;
- optional OpenAI-compatible embedding provider;
- deterministic Reciprocal Rank Fusion (RRF);
- independent dense, lexical and hybrid retrieval modes.

Exit condition: unit tests require no network/model download and adapters are swappable without changing CT-RAG semantics.

### #3 — Causal path confidence and provenance weighting

Deliverables:

- provenance-calibrated causal confidence;
- best path reconstruction, not only scalar confidence;
- ancestor/descendant traversal budgets;
- cycle-safe multipath traversal;
- causal evidence exposed on `RetrievalHit`;
- explicit exclusion of temporal/behavioral edges from causal confidence.

Exit condition: equivalent inferred evidence ranks below observed execution evidence and the selected path is explainable.

### #4 — Basin and attractor engine

Deliverables:

- explicit attractor descriptors with metadata/confidence/origin;
- basin membership API;
- neighboring basin API;
- basin boundary API;
- explainable shared-basin scoring;
- branching/converging/cyclic tests.

Exit condition: unrelated graph components cannot leak into basin results and all basin relations can be inspected programmatically.

### #5 — Event Sourcing projector and NDJSON contracts

Deliverables:

- validated `EventRecord` contract;
- configurable external field mappings;
- idempotent duplicate replay;
- explicit event identity/error reporting;
- deterministic projection of causation vs temporal/behavioral sequence;
- multi-step integration fixture.

Exit condition: replaying the same event stream produces the same topology without duplicate nodes/edges.

### #6 — Staged WHY / WHAT_NEXT / RECOVERY / COUNTERFACTUAL retrieval

Deliverables:

- explicit staged pipeline: anchor discovery -> topology/basin expansion -> causal traversal -> reranking;
- direction-specific behavior per query mode;
- recovery trajectory retrieval;
- observational divergence retrieval for counterfactual support;
- score/path/anchor explanations in results.

Exit condition: one deterministic graph demonstrates materially different retrieval behavior for each query mode.

### #7 — Benchmark and ablation harness

Already implemented baseline:

- lexical-only;
- dense-only;
- dense+lexical;
- graph/topology-only;
- dense+causal;
- dense+topological;
- full CT-RAG;
- Recall@K, Precision@K, MRR, nDCG, causal/path/trajectory/basin/recovery/context metrics;
- deterministic seeds/configuration;
- JSON/CSV/table output;
- CI artifacts.

Remaining closure work:

- reconcile the committed harness against every acceptance criterion;
- ensure documentation accurately states that current dense retrieval is a hashing proxy and current lexical retrieval is not BM25;
- retain the controlled validation in `REPORT.md` without overstating external validity.

Exit condition: one command reproduces all arms and issue #7 points to reproducible CI evidence.

### #8 — Dynamic terrain reinforcement, erosion and attractor discovery

Deliverables:

- transition frequency tracking;
- navigation weight separate from historical edge evidence;
- configurable reinforcement and time decay;
- strongly connected components;
- sink/recurrent-state discovery;
- empirically discovered attractors distinguishable from manually declared ones;
- basin snapshot/drift metric.

Exit condition: repeated trajectories strengthen expected paths, decay changes navigation without deleting evidence, and fixed data/config yields deterministic attractors.

### #9 — Persistence interfaces and local scalable adapter

Deliverables:

- `MemoryStore`;
- `VectorIndex`;
- `TopologyStore`;
- `EventSource`;
- `TerrainStore`;
- in-memory conformance implementations;
- SQLite persistent implementation for the complete semantic state required by CT-RAG;
- adapter equivalence tests.

Exit condition: save/reload preserves all ranking inputs and produces equivalent retrieval outputs.

### #10 — Research visualization, reproducibility and paper-ready artifacts

Deliverables:

- formal problem statement and notation;
- machine-generated terrain/basin/causal-path figures;
- benchmark/ablation tables generated from result files;
- experiment manifest/config linkage;
- sourced related work, including BasinRAG;
- explicit observed/inferred/hypothesized/counterfactual distinctions;
- README research-artifact index.

Exit condition: figures/tables are regenerable from machine-readable experiment output and every reported benchmark result has a reproducible command/config.

## v0.1 definition of done

v0.1 is complete when all ten issues are closed and a fresh clone can:

```bash
pip install -e ".[dev]"
pytest
python -m ctrag.benchmarks
```

and can additionally:

- ingest and replay an event trace idempotently;
- preserve explicit causal provenance;
- retrieve WHY/WHAT_NEXT/RECOVERY/COUNTERFACTUAL evidence with explanations;
- inspect basins/attractors;
- persist/reload retrieval state;
- reproduce benchmark tables and research visualizations.

## Research validation beyond v0.1

The controlled synthetic result in `REPORT.md` is evidence that the architecture works for authored event-sourced traces. It is not yet evidence of universal superiority.

Post-v0.1 external validation should add:

- real event-sourced traces;
- strong BM25 baseline;
- learned dense embedding baselines;
- GraphRAG/BasinRAG comparisons under matched datasets;
- anchor-discovery evaluation rather than known anchors only;
- prospective prediction tasks without future-state visibility;
- larger graphs and independent datasets;
- confidence intervals and preregistered statistical comparisons.
