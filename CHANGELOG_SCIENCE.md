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

## SCI-003 — No-oracle anchor discovery degrades end-to-end retrieval

**Date:** 2026-09-09  
**Issue:** #17  
**Classification:** `negative_result`  
**Status:** retained as a limitation before final-holdout unblinding  
**Code changed because of the result:** `false`

### Observation

The oracle-anchor proof-of-concept does not translate directly into equivalent end-to-end performance when the current execution state must be selected from raw query text.

On run `34353826174`, hybrid selection at K=3 changed from:

| Metric | Oracle anchor | Discovered Top-1 |
| --- | ---: | ---: |
| Recall@3 | 0.9583 | 0.4803 |
| nDCG@3 | 0.9710 | 0.4507 |
| Causal Path Recall | 0.9167 | 0.4306 |

The strongest observed Top-1 anchor accuracy in the failure/recovery dataset was lexical selection at `0.5000`. Propagating Top-3 anchor uncertainty recovered part of the ranking loss but did not restore oracle causal-path performance.

### Scientific impact

The original oracle-anchor benchmark remains useful for isolating causal/topological navigation, but it is not an end-to-end agent-memory score. Anchor selection is currently a first-order error source and must remain separately reported.

No retrieval implementation was changed to turn this result into a win. The final holdout remains sealed.

Detailed evidence: `docs/ANCHOR_DISCOVERY_01.md`.

---

## SCI-004 — Synthetic trace construction is adversarial to semantic-only graph retrieval

**Date:** 2026-09-09  
**Issue:** #21  
**Classification:** `dataset_artifact`  
**Status:** documented before external-validity claims  
**Code changed because of the observation:** `false`

### Observation

The matched semantic GraphRAG-style comparator obtains `0.0000` Recall@3/nDCG@3/Causal Path Recall on the controlled `dev` split, while Full CT-RAG remains high.

The comparator was not intentionally deprived of semantic retrieval: it receives the same learned embedding model, BM25 component, corpus, oracle anchor and K values. It is intentionally denied CT-RAG execution-causal edges because those are the signal under test.

The synthetic corpus repeats semantically similar event descriptions across distinct execution traces while the relevance target is trace-specific. A semantic k-NN relation graph therefore connects textually similar states but lacks evidence identifying which execution produced the target state.

### Scientific impact

This is useful mechanism evidence that semantic graph connectivity is not equivalent to observed execution causality. It is **not** evidence that Microsoft GraphRAG or graph retrieval generally has zero utility.

The zero baseline score must be treated as a property of this synthetic mechanism test until real/independent datasets are evaluated.

Detailed evidence: `docs/GRAPHRAG_COMPARISON_01.md`.

---

## SCI-005 — BasinRAG reproduction artifact serialization failure

**Date:** 2026-09-09  
**Issue:** #22  
**Classification:** `correctness_bug`  
**Status:** corrected, pending post-fix science rerun  
**Final holdout:** sealed

### Defect

Run `34355758910` successfully installed and imported the pinned upstream `Basinfy/BasinRAG@fb62771eda11f1a70d6e99ca7aa5e19b9825b829`, passed the unit/science gates, and repeatedly built BasinRAG functional attraction basins. The run then failed while serializing `costs.csv`.

Index-cost rows do not have `query_id`; query-cost rows do. The helper derived CSV fieldnames from only the first row, causing:

```text
ValueError: dict contains fields not in fieldnames: 'query_id'
```

This is an output-artifact defect in our reproduction harness, not a BasinRAG retrieval/topology failure and not a negative CT-RAG result.

### Regression and correction

Regression test:

```text
tests/test_basinrag_reproduction.py::test_csv_handles_mixed_index_and_query_cost_rows
```

The CSV schema is now the deterministic union of keys appearing across all rows. Missing values are serialized as empty cells.

No query, relevance label, topology edge, model revision, K, token budget, dataset split or retrieval score was changed by this correction.

### Scientific impact

The failed run is retained as pre-fix evidence. Comparison claims from #22 remain pending until the corrected train/dev reproduction completes and its artifacts are inspected. The final holdout remains sealed.

---

## How future entries must be recorded

Each future scientific change must receive an ID (`SCI-006`, `SCI-007`, ...), an entry in `research/science-changes.json`, and one section in this file.

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
