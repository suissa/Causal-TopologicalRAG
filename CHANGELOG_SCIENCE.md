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

## How future entries must be recorded

Each future scientific change must receive an ID (`SCI-002`, `SCI-003`, ...), an entry in `research/science-changes.json`, and one section in this file.

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
