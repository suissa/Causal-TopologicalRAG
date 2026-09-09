# Upstream BasinRAG Matched Comparison — Experiment 01

**Issue:** #22  
**Successful reproduction run:** `34356737770`  
**CT-RAG commit evaluated:** `5eaef698c19260038f98e936e7b8705072d4f66e`  
**BasinRAG upstream:** `Basinfy/BasinRAG@fb62771eda11f1a70d6e99ca7aa5e19b9825b829`  
**BasinRAG version:** `1.0.3`  
**Evaluation split:** `dev` only  
**Final holdout:** sealed  
**Acceptance status:** complete after current-head science-gate validation

## Purpose

This experiment tests whether CT-RAG adds measurable information beyond a real topological retrieval implementation based on dynamical basins of attraction.

Unlike `docs/GRAPHRAG_COMPARISON_01.md`, this is not an in-house surrogate for the comparison target. The workflow installs the pinned upstream BasinRAG Git commit and invokes its own:

- `BasinTopologyEngine`;
- `build_graph`;
- `partition_into_basins`;
- `build_meta_basins`;
- `BM25Index`;
- `TopologicalLocalSearch`;
- `HybridSearch`.

The upstream installation and import are validated in CI before the comparison runs.

## Matched protocol

The matched comparison uses:

- identical synthetic event text corpus;
- identical raw query text for end-to-end arms;
- K = `3`;
- context token budget = `64`;
- `sentence-transformers/all-MiniLM-L6-v2`;
- pinned encoder revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- 384-dimensional normalized embeddings.

BasinRAG receives document/source sequence and can construct its own sequential/semantic functional topology. It receives **no CT-RAG causal edges, causation labels or gold relevance labels**.

The upstream cross-encoder reranker is disabled in this matched mechanism experiment so that the comparison isolates retrieval/topology rather than the quality of a second learned reranker. This means the result is a comparison against the real upstream BasinRAG topology + hybrid retrieval components, not a reproduction of every default production-stage component.

## Compared arms

1. `basinrag_upstream_hybrid`
   - real pinned BasinRAG functional topology/basins;
   - upstream BM25 + dense + HybridSearch;
   - raw query only.

2. `ctrag_full_no_oracle`
   - CT-RAG with explicit execution-causal topology;
   - no gold anchor;
   - top-3 anchors discovered from the raw query.

3. `ctrag_no_causal_metadata`
   - same CT-RAG retrieval machinery;
   - every explicit `CAUSAL` edge removed before retrieval;
   - measures how much survives from semantic/temporal/behavioral structure alone.

4. `ctrag_oracle_diagnostic`
   - exact gold anchor supplied;
   - diagnostic upper-bound/mechanism track only;
   - must never be merged into end-to-end claims.

## Aggregate dev result at K=3

Across 72 query instances:

| Arm | Recall@3 | MRR | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Trajectory reconstruction | Context efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BasinRAG upstream hybrid | 0.0208 | 0.0278 | 0.0077 | 0.0000 | 36.000 | 0.2500 | 0.0192 |
| CT-RAG no causal metadata | 0.4005 | 0.4792 | 0.2967 | 0.2014 | 26.958 | 0.4958 | 0.3664 |
| **CT-RAG full, no oracle** | **0.6146** | **0.7153** | **0.6676** | **0.3750** | **14.625** | **0.6750** | **0.5429** |
| CT-RAG oracle diagnostic | 0.9583 | 1.0000 | 0.9710 | 0.9167 | 1.500 | 0.9000 | 0.8819 |

Two different effects are visible:

- explicit causal information substantially improves the same CT-RAG retrieval machinery over its no-causal ablation;
- removing the oracle anchor still leaves a large gap to the diagnostic upper-bound, consistent with the independent #17 anchor-discovery experiment.

## Causal contribution ablation

Comparing the two **non-oracle CT-RAG** arms at K=3:

| Metric | No causal metadata | Full causal CT-RAG | Change |
| --- | ---: | ---: | ---: |
| Recall@3 | 0.4005 | 0.6146 | +0.2141 |
| nDCG@3 | 0.2967 | 0.6676 | +0.3709 |
| Causal Path Recall | 0.2014 | 0.3750 | +0.1736 |
| Causal Distance Error ↓ | 26.958 | 14.625 | -12.333 |
| Trajectory reconstruction | 0.4958 | 0.6750 | +0.1792 |
| Context efficiency | 0.3664 | 0.5429 | +0.1765 |

This is the strongest mechanism result from the matched comparison: when the retrieval implementation, corpus, raw queries and encoder are held fixed, removing explicit causal edges materially degrades the execution-oriented retrieval metrics on the controlled dataset.

It is still synthetic internal-validity evidence; statistical inference and external datasets remain separate gates.

## By dataset

### Failure/recovery

| Arm | Recall@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ |
| --- | ---: | ---: | ---: | ---: |
| BasinRAG upstream | 0.0000 | 0.0000 | 0.0000 | 36.0 |
| CT-RAG no causal | 0.5602 | 0.3606 | 0.3056 | 24.5 |
| **CT-RAG full no-oracle** | **0.7917** | **0.8104** | **0.5833** | **7.5** |
| Oracle diagnostic | 1.0000 | 1.0000 | 1.0000 | 0.0 |

### Branching

| Arm | Recall@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ |
| --- | ---: | ---: | ---: | ---: |
| BasinRAG upstream | 0.0417 | 0.0155 | 0.0000 | 36.0 |
| CT-RAG no causal | 0.2407 | 0.2328 | 0.0972 | 29.42 |
| **CT-RAG full no-oracle** | **0.4375** | **0.5247** | **0.1667** | **21.75** |
| Oracle diagnostic | 0.9167 | 0.9420 | 0.8333 | 3.0 |

Branching remains much harder than linear failure/recovery trajectories, especially without a reliable exact anchor.

## By query mode

| Mode | BasinRAG Recall@3 | CT no-causal Recall@3 | CT full no-oracle Recall@3 | Oracle Recall@3 |
| --- | ---: | ---: | ---: | ---: |
| WHY | 0.0000 | 0.1574 | **0.6250** | 1.0000 |
| RECOVERY | 0.0000 | 0.6389 | **0.7500** | 1.0000 |
| WHAT_NEXT | 0.0625 | **0.6458** | 0.5313 | 0.8750 |

The `WHAT_NEXT` result is important: the no-causal CT-RAG arm has higher Recall@3 than full CT-RAG in this aggregate, although full CT-RAG has better nDCG in the failure/recovery subset. Therefore explicit causality is **not uniformly beneficial for every query mode/metric** under the current no-oracle ranking policy. This result is preserved rather than retuned away.

## Resource measurements

Mean measured query latency on the tiny `dev` graphs:

| Arm | Mean ms/query |
| --- | ---: |
| BasinRAG upstream hybrid | 10.002 |
| CT-RAG no causal metadata | 13.239 |
| CT-RAG full no-oracle | 13.697 |

BasinRAG index construction averaged approximately `426.15 ms` per generated dataset/seed in this runner.

These values are useful for matched small-graph accounting only. They are not scaling evidence; #29 owns scale-dependent conclusions.

## What this experiment demonstrates

Supported on this controlled train/dev protocol:

- the actual pinned BasinRAG 1.0.3 topology can be reproduced inside the CT-RAG scientific harness;
- BasinRAG's sequential/semantic basin topology is not equivalent to explicit execution-causal topology on these trace-specific tasks;
- within CT-RAG itself, removing causal metadata produces a substantial aggregate degradation on causal/trajectory metrics;
- CT-RAG still suffers a major no-oracle anchor gap;
- explicit causality is not uniformly better for every query mode, as shown by `WHAT_NEXT` Recall@3.

## What this experiment does not demonstrate

It does **not** prove:

- CT-RAG is universally superior to BasinRAG;
- BasinRAG performs poorly on its intended long-document retrieval tasks;
- the published BasinRAG SciFact/SWE-bench numbers are false;
- the full default BasinRAG pipeline with cross-encoder reranking would have these same scores;
- external validity on natural event-sourced systems;
- causal identification from observational chronology alone.

The controlled corpus was authored around execution traces and is therefore much closer to CT-RAG's intended information structure than to BasinRAG's long-document use case. That asymmetry is a limitation and is why Gates C/D require independent data and falsification controls before headline scientific claims.

## Reproduction

The science workflow installs:

```bash
pip install "git+https://github.com/Basinfy/BasinRAG.git@fb62771eda11f1a70d6e99ca7aa5e19b9825b829"
```

and executes:

```bash
python -m ctrag.benchmarks.basinrag_reproduction train \
  --output basinrag-results/train

python -m ctrag.benchmarks.basinrag_reproduction dev \
  --output basinrag-results/dev
```

The workflow also fingerprints the sealed final test after these experiments. Run `34356737770` completed that check successfully.
