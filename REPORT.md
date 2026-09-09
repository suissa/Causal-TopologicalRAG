# CT-RAG Experimental Validation Report

**Project:** Causal-Topological RAG (CT-RAG)  
**Status:** controlled proof-of-concept evidence, under scientific validation  
**Validated commit:** `593bdd67ffe802397f50738de6fb320a93f1a178`  
**CI run:** https://github.com/suissa/Causal-TopologicalRAG/actions/runs/34344883091  
**Python artifact used for the numbers below:** `benchmarks-python-3.12`  
**Artifact SHA-256:** `671a87326d293939915cb00f2588bc9bf5ae86243385a5d178bf22eda59c8d3f`

## Executive conclusion

The current experiment provides positive empirical evidence that CT-RAG can exploit explicit causal/topological structure on the controlled synthetic event-sourced tasks implemented in this repository.

It does **not** establish external superiority over production RAG systems. The current benchmark is synthetic, retrospective, gives all retrieval arms the gold/oracle anchor, uses a deterministic hashing embedding proxy rather than a learned dense model, and uses IDF overlap rather than a production BM25 baseline. These limitations are now tracked by the scientific validation issues.

The first scientific audit also found and corrected a real retrieval bug: the historical `search()` implementation treated `RECOVERY` and `COUNTERFACTUAL` causal traversal as bidirectional. That contradicted the query-mode contract. `RECOVERY` must retrieve causal descendants; observational counterfactual/divergence support must retrieve causal ancestors. Regression tests were added before the fix. The intentionally failing pre-fix CI produced `87 passed, 2 failed`; after the correction the same suite produced `89 passed` on Python 3.12, and all CI jobs passed on Python 3.11, 3.12 and 3.13.

Because this correction materially changes benchmark outputs, the earlier report numbers are superseded by this file.

## 1. Corrected K=3 result

The benchmark contains 72 query instances, 7 retrieval arms and 4 K values, yielding 2,016 observations.

At `K=3`:

| System | Recall@3 | MRR@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Context-token efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CT-RAG** | **0.9583** | **1.0000** | **1.0000** | **0.9167** | **1.5000** | **0.8819** |
| Dense + Causal | 0.6019 | 0.8287 | 0.7632 | 0.3333 | 14.3333 | 0.5206 |
| Graph / Topology only | 0.7500 | 0.8588 | 0.6331 | 0.4792 | 13.8333 | 0.7230 |
| Dense + Topological | 0.3021 | 0.3773 | 0.2376 | 0.0833 | 29.5833 | 0.3276 |
| Dense only | 0.0428 | 0.1273 | 0.0292 | 0.0000 | 35.8333 | 0.0455 |
| Dense + Lexical | 0.0382 | 0.1319 | 0.0255 | 0.0000 | 36.0000 | 0.0388 |
| Lexical only | 0.0278 | 0.0787 | 0.0161 | 0.0000 | 36.0000 | 0.0256 |

Relative to graph/topology-only at `K=3`, full CT-RAG currently shows:

- Recall: **+27.8%** relative improvement;
- nDCG: **+58.0%**;
- Causal Path Recall: **+91.3%**;
- context-token efficiency: **+22.0%**;
- Causal Distance Error: **89.2% lower**.

These effect sizes are descriptive for the current synthetic dataset. Formal confidence intervals and preregistered paired inference belong to scientific issue #25.

## 2. Why the audit correction changed the result

Before the audit, the historical retriever used both incoming and outgoing causal paths for `RECOVERY` and `COUNTERFACTUAL`.

That was inconsistent with the intended semantics:

```text
WHY             -> causal ancestors
COUNTERFACTUAL  -> historical ancestor/divergence support
WHAT_NEXT       -> causal descendants
RECOVERY        -> causal descendants / recovery trajectory
SIMILAR         -> bidirectional structural context allowed
```

The audit added regression tests that first failed against the old behavior. In the failure fixture, an ancestor of a failure received causal score `1.0` during a `RECOVERY` query, and a consequence received causal score `1.0` during a `COUNTERFACTUAL` query. Those are directionally invalid signals.

The correction removed those signals without changing:

- query labels;
- causal edges;
- benchmark seeds;
- retrieval weights;
- K values;
- candidate corpus;
- metric definitions.

The largest impact is on `RECOVERY`, where the corrected Full CT-RAG result at `K=3` becomes:

```text
Recall@3                 = 1.0000
MRR@3                    = 1.0000
nDCG@3                   = 1.0000
Causal Path Recall       = 1.0000
Causal Distance Error    = 0.0000
Recovery Path Precision  = 1.0000
Context-token efficiency = 1.0000
```

This improvement is therefore a correctness correction, not a post-hoc parameter optimization.

## 3. Per-mode corrected result

### WHY, K=3

Full CT-RAG:

```text
Recall@3                       = 1.0000
MRR@3                          = 1.0000
nDCG@3                         = 1.0000
Causal Path Recall             = 1.0000
Causal Distance Error          = 0.0000
Trajectory Reconstruction      = 0.9167
Context-token efficiency       = 0.8750
```

`WHY` remains the strongest controlled demonstration of the core hypothesis: when the task is to retrieve explicit causal ancestors, semantic lookalikes alone are insufficient in these fixtures.

### RECOVERY, K=3

Full CT-RAG:

```text
Recall@3                       = 1.0000
MRR@3                          = 1.0000
nDCG@3                         = 1.0000
Causal Path Recall             = 1.0000
Causal Distance Error          = 0.0000
Trajectory Reconstruction      = 1.0000
Recovery Path Precision        = 1.0000
Context-token efficiency       = 1.0000
```

### WHAT_NEXT, K=3

Full CT-RAG:

```text
Recall@3                       = 0.8750
MRR@3                          = 1.0000
nDCG@3                         = 1.0000
Causal Path Recall             = 0.7500
Causal Distance Error          = 4.5000
Trajectory Reconstruction      = 0.7500
Context-token efficiency       = 0.8333
```

The non-perfect `WHAT_NEXT` result is useful: the synthetic benchmark is not universally saturated at `K=3`.

## 4. Adversarial and leakage controls added by the audit

The scientific audit introduced controls that explicitly verify:

1. a semantic lookalike connected only temporally does not receive causal evidence;
2. a temporal chain cannot satisfy a causal-path metric;
3. retrieval in the wrong direction cannot satisfy a `WHY` causal path;
4. opaque anchor IDs are not present in natural-language query text;
5. relevant/gold node IDs are not exposed in query text;
6. the oracle anchor is excluded from the relevance target;
7. gold IDs correspond to actual topology nodes;
8. hand-calculated empty and partial rankings preserve metric contracts.

These are necessary internal-validity controls, but they are not a substitute for independent external datasets.

## 5. What the corrected experiment supports

Within the current synthetic, retrospective, oracle-anchor benchmark, the evidence supports:

1. explicit causal/topological structure provides retrieval information not captured by the current semantic/lexical proxies alone;
2. directional causal traversal matters to retrieval correctness;
3. combining causal, topological, semantic and behavioral signals can improve small-context retrieval on the authored execution tasks;
4. `WHY` causal-ancestor retrieval is strongly recovered by CT-RAG in these fixtures;
5. corrected `RECOVERY` traversal reconstructs the authored descendant recovery paths at `K=3`;
6. causal/topological retrieval can substantially increase relevant-context density in this controlled setting;
7. the corrected code and benchmark execute successfully on Python 3.11, 3.12 and 3.13.

## 6. What is still unsupported

This experiment does **not** establish that CT-RAG:

- beats learned dense retrieval on independent real data;
- beats production BM25;
- beats GraphRAG under a matched protocol;
- beats BasinRAG under a matched protocol;
- works without a known/oracle anchor;
- generalizes beyond the authored trace templates;
- predicts future events without future-state leakage;
- identifies counterfactual/interventional causal effects from ordinary event logs;
- remains superior under missing or corrupted causal metadata;
- scales economically to very large event graphs;
- improves downstream LLM answer quality on independent tasks.

These are explicit targets of the Phase 2 scientific validation program (#14–#35).

## 7. Reproducibility

Corrected validation command:

```bash
pip install -e ".[dev]"
pytest
python -m ctrag.benchmarks
python -m ctrag.research_artifacts \
  --benchmark-dir benchmark-results \
  --out research-artifacts \
  --k 3
```

Validated CI evidence:

```text
Python 3.12: 89 passed in 0.97s
Benchmark:   2016 query/K/baseline observations
Research artifacts: 4 generated files
```

The Python 3.12 benchmark artifact is identified by:

```text
GitHub Actions artifact ID: 10101257954
SHA-256: 671a87326d293939915cb00f2588bc9bf5ae86243385a5d178bf22eda59c8d3f
```

## 8. Scientific status

The appropriate claim at this stage is:

> CT-RAG has a reproducible positive proof-of-concept on controlled event-sourced synthetic traces, and the first adversarial audit found a directional retrieval bug whose correction strengthened rather than weakened the controlled result.

The next objective is **falsification and external validation**, not optimization against this benchmark. The result should be considered externally defensible only after preregistration, frozen holdout evaluation, strong BM25/learned-dense comparisons, no-oracle anchor evaluation, independent datasets, topology-destruction controls, uncertainty estimates and clean-room replication.
