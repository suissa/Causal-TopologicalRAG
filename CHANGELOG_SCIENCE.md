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
**Status:** corrected and validated before final-holdout unblinding  
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

### Post-fix evidence

Validation run `34357104431` completed successfully. It passed the regression suite and science gate, reran strong baselines, no-oracle anchor evaluation, matched GraphRAG, and the pinned BasinRAG reproduction on both `train` and `dev`, verified the final holdout remained sealed, and uploaded all scientific evidence artifacts.

The BasinRAG reproduction artifact produced by that run has digest:

```text
sha256:877e22478c7cd8a23919c7c232b5b76eb29bdbbeedc457542728be309e7192ab
```

### Scientific impact

The failed run remains preserved as pre-fix evidence, while run `34357104431` is the post-fix validation evidence. The serialization defect is therefore closed and does not invalidate the matched BasinRAG comparison. The final holdout remained sealed throughout.

Detailed comparison evidence: `docs/BASINRAG_COMPARISON_01.md`.

---

## SCI-006 — Gate D effects are query-specific and include null/adverse results

**Date:** 2026-09-13  
**Issues:** #23, #24, #27, #30  
**Classification:** `negative_result`  
**Status:** retained before final-holdout unblinding  
**Code changed because of the observation:** `false`

The complete, pre-specified Gate D grid was retained. Removing causal edges reduced WHY Recall@3 from 1.000 to 0 and eliminated causal evidence in every mode, but increased RECOVERY Recall@3 from 0.667 to 1.000. Removing temporal edges changed none of the headline metrics. Random hypothesized-edge noise produced little aggregate degradation in this small synthetic graph.

These results reject the universal claim that more causal/topological structure always improves retrieval. The supported claim is narrower: valid direction and endpoints materially support WHY and some WHAT_NEXT retrieval, while the current RECOVERY weighting can be harmed by causal structure. Noise robustness on larger or external causal graphs remains unproven.

No retrieval weights, relevance labels, K values, queries, preregistration bytes, or frozen holdout bytes were changed in response. The implementation of the already-planned falsification harness is separate from this observation. Detailed evidence: `docs/GATE_D_FALSIFICATION.md` and `research/gate-d-v1/gate-d-results.json`.

---

## SCI-007 — Supplied interventional evidence is not discovery or identification

**Date:** 2026-09-13  
**Issues:** #38, #41  
**Classification:** `negative_result`  
**Status:** retained before final-holdout unblinding  
**Code changed because of the observation:** `false`

Across three fixed-equation SCM families and five seeds, CT-RAG with supplied true causal topology achieved intervention-pair Recall@2 of 1.000, versus 0.000 for dense retrieval. This supports causal-evidence navigation under valid supplied topology.

CT-RAG did not discover the graph and emitted no individual counterfactual estimate: counterfactual coverage is 0.0. The hidden-confounder negative control also retained the naive observational ATE error of approximately 1.806. These results prohibit relabelling retrieval as causal discovery or counterfactual identification.

No weights, labels or mechanisms were changed after observing the result. The frozen final holdout remains sealed. Detailed evidence: `docs/SCM_GROUND_TRUTH_01.md`.

---

## SCI-008 — Root cause is an anomalous causal frontier, not merely a nearby ancestor

**Date:** 2026-09-14
**Classification:** `modeling_assumption`
**Status:** exploratory; requires a new frozen validation set
**Final holdout:** sealed

The first commercial-system experiment defines operational root cause as the earliest anomalous observation on an evidenced path to a selected symptom: the candidate is anomalous and has no anomalous causal ancestor. This definition is computed from public event status and `causation_id` topology; oracle role labels and answer IDs remain outside retrieval.

Across six designed incidents, generic `full_ctrag` did not rank the root cause in Top-3 (MRR 0.228). It correctly favors nearby causal ancestors but does not by itself distinguish an immediate cause from the first anomaly. The task-specific `ctrag_causal_frontier` operator achieved Top-1 1.000, and `full_ctrag` RECOVERY retrieved each observed solution at Top-1.

The frontier operator was designed and evaluated during this same exploratory experiment, so 6/6 is implementation evidence rather than an unbiased generalization estimate. A future preregistered/frozen incident set must test unseen topologies, missing/incorrect causal edges, multiple simultaneous root causes, warning-status noise and unobserved causes. No final-holdout data was loaded.

Detailed evidence: `research/commercial-system-v1/REPORT.md`, `results.json`, `oracle.json` and the self-contained `explorer.html`.

---

## SCI-009 — Independent evidence signals can produce an inferred causal hypothesis

**Date:** 2026-09-14
**Classification:** `modeling_assumption`
**Status:** exploratory; inferred edge requires later validation
**Final holdout:** sealed

The commercial experiment now includes a seventh incident in which the configuration drift, metric anomaly, trace mismatch, log evidence and sales symptom have no `step.parent` and therefore no runtime-declared causal edge. CT-RAG derives lower-confidence `INFERRED` edges only from a shared evidence group, signal diversity and ordering. The cause hypothesis is ranked using independent-signal support rather than direct parentage.

The inferred scenario is recovered at Top-1 by `ctrag_causal_frontier`; all seven scenarios retain Recall@3 of 1.000 for frontier diagnosis and `full_ctrag` recovery. This is not causal proof: shared evidence groups can be confounded, and the inferred relation must be validated against unseen incidents, randomized perturbations or authoritative runtime evidence.

Detailed evidence: `research/commercial-system-v1/REPORT.md`, `graph.json`, `results.json` and `oracle.json`.

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
