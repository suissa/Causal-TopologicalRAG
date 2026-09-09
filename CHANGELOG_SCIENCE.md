# Scientific Change Log

This file records post-v0.1 changes that can affect scientific interpretation. It is deliberately separate from the software changelog.

A scientific result is allowed to get worse. A loss, null effect or surprising result is not classified as a bug merely because CT-RAG underperforms.

Allowed classifications are:

```text
correctness_bug
modeling_assumption
dataset_artifact
negative_result
protocol_amendment
exploratory_analysis
```

For `correctness_bug`, the required order is:

```text
observe anomaly
  -> reproduce with smallest failing regression test
  -> classify
  -> fix on train/dev only
  -> rerun affected analyses
  -> preserve before/after evidence
```

The final holdout must remain sealed while correctness changes are being resolved. After the final test has been unblinded, a new correctness change cannot retroactively restore pristine-holdout status; subsequent test runs are replications.

---

## SCI-001 — Query-mode causal direction bug

**Date:** 2026-09-09  
**Issue:** #14  
**Classification:** `correctness_bug`  
**Status:** corrected before Phase 2 preregistration and final-holdout freeze

### Defect

The historical `search()` implementation gave `RECOVERY` and `COUNTERFACTUAL` causal score in both graph directions.

Correct semantics are:

```text
WHY             -> ancestors
COUNTERFACTUAL  -> ancestors / observed historical divergence support
WHAT_NEXT       -> descendants
RECOVERY        -> descendants / historical recovery trajectory
```

### Regression-before-fix evidence

Regression tests were added before the implementation change:

```text
test_recovery_causal_signal_is_descendant_only_regression
test_counterfactual_causal_signal_is_ancestor_only_regression
```

Pre-fix CI run `34344592917`:

```text
87 passed, 2 failed
```

The two failures were exactly the newly exposed wrong-direction causal signals.

### Correction

A single query-mode direction policy was introduced and exported through the public `CTRetriever`. The benchmark runner was switched to the corrected implementation.

No gold labels, graph edges, seeds, K values, retrieval weights or metric formulas were changed to restore performance.

### Post-fix evidence

Post-fix CI run `34344883091`:

```text
89 passed on Python 3.12
CI green on Python 3.11 / 3.12 / 3.13
2016 benchmark observations regenerated
```

Full CT-RAG at K=3 changed from:

| Metric | Before | After |
| --- | ---: | ---: |
| Recall@3 | 0.9028 | 0.9583 |
| nDCG@3 | 0.9745 | 1.0000 |
| Causal Path Recall | 0.7500 | 0.9167 |
| Causal Distance Error ↓ | 3.5000 | 1.5000 |
| Context-token efficiency | 0.8365 | 0.8819 |

The corrected result is stronger, but that fact is incidental to the classification: this is a bug because the implementation contradicted the declared query semantics, not because the metric improved.

### Scientific impact

`REPORT.md` was superseded with corrected numbers and explicitly retains the narrower proof-of-concept claim. `docs/SCIENTIFIC_AUDIT_01.md` records the audit in full.

---

## SCI-002 — Strong-baseline query-cost cache contamination

**Date:** 2026-09-09  
**Issue:** #16  
**Classification:** `correctness_bug`  
**Status:** corrected before final-holdout unblinding

### Defect

The first strong-baseline run correctly produced retrieval rankings, but its resource-cost protocol was not fair. The exact query embedding could remain in `CachedEmbedder` after the dense arm and then be reused by hybrid and Full CT-RAG. The hybrid timing also fused a dense ranking computed before its own timed block.

Therefore the first-run latency values represented mixed warm-cache work and could not be used for cross-arm performance claims. This defect did **not** change retrieved IDs, relevance metrics, causal metrics or the frozen dataset.

### Regression and correction

The regression test:

```text
tests/test_strong_baselines.py::test_cached_embedder_invalidation_forces_fresh_query_embedding
```

proves that invalidating a query forces a new embedding computation.

The benchmark now invalidates the exact query embedding before every learned arm and recomputes dense retrieval inside the hybrid arm's timed block. Document embeddings remain indexed, which models normal retrieval operation rather than re-indexing the corpus for every query.

### Before/after evidence

Pre-fix resource run: `34352869207`. Its warm-cache query times are retained as invalid historical evidence and are not used as final resource measurements.

Corrected run: `34353826174`.

Mean cold query latency on the synthetic `dev` split:

| Arm | Mean ms/query |
| --- | ---: |
| BM25 (`rank-bm25`) | 1.003 |
| MiniLM dense | 11.494 |
| MiniLM + BM25 hybrid | 11.787 |
| Full CT-RAG + MiniLM/BM25 | 12.652 |
| MPNet dense | 45.578 |
| MPNet + BM25 hybrid | 45.892 |
| Full CT-RAG + MPNet/BM25 | 46.926 |

Retrieval-quality aggregates were unchanged by this correction. The final test remained sealed throughout.

### Scientific impact

Quality conclusions from the first strong-baseline run remain valid for the same synthetic `dev` split, but all resource comparisons must use the corrected run `34353826174` or later.

---

## How future entries must be recorded

Each future scientific change must receive an ID (`SCI-003`, `SCI-004`, ...), an entry in `research/science-changes.json`, and one section in this file.

A `correctness_bug` entry must include:

- linked issue/defect;
- regression-test path(s);
- evidence before the fix;
- evidence after the fix;
- affected files/claims/metrics;
- whether the final holdout was still sealed.

A `negative_result` entry must explicitly state `code_changed=false` unless a separate independently demonstrated correctness bug exists. Negative results remain in final reporting.

A `modeling_assumption` or `dataset_artifact` may motivate a new exploratory experiment, but it cannot be silently rewritten into a correctness defect.

A `protocol_amendment` requires a new protocol version/fingerprint before the affected final result is viewed.
