# Scientific Audit 01 — Internal Validity of the CT-RAG Synthetic Benchmark

**Issue:** #14  
**Date:** 2026-09-09  
**Scope:** dataset generation, labels, query construction, anchor policy, retrieval direction, metrics, adversarial controls and benchmark claims.

## Audit objective

The purpose of this audit was to try to falsify the current controlled result before adding stronger baselines or independent datasets.

The audit policy was intentionally asymmetric: a negative result would be retained as scientific evidence; implementation changes were allowed only when a concrete correctness defect could first be reproduced by a failing regression test.

## Finding A1 — confirmed correctness bug

### Historical behavior

The v0.1 `search()` implementation treated causal traversal as:

```text
WHY       -> incoming
WHAT_NEXT -> outgoing
all other modes -> incoming + outgoing
```

This meant `RECOVERY` and `COUNTERFACTUAL` received causal score from both ancestors and descendants.

### Why this is incorrect

The public query contract already defines:

```text
WHY             -> causal ancestors
COUNTERFACTUAL  -> historical ancestor/divergence support
WHAT_NEXT       -> causal descendants
RECOVERY        -> causal descendants / recovery trajectory
SIMILAR         -> bidirectional structural context is allowed
```

An ancestor of the current failure is not a recovery step. Likewise, a consequence occurring after the current state is not an earlier divergence point.

### Regression evidence before correction

Two tests were committed before the implementation fix:

- `test_recovery_causal_signal_is_descendant_only_regression`
- `test_counterfactual_causal_signal_is_ancestor_only_regression`

The pre-fix CI produced:

```text
87 passed, 2 failed
```

Both failures showed an invalid causal component of `1.0` in the wrong direction.

### Correction

The corrected direction policy is:

```text
WHY             -> in
COUNTERFACTUAL  -> in
WHAT_NEXT       -> out
RECOVERY        -> out
SIMILAR         -> in + out
```

No labels, weights, graph edges, seeds, K values or metric formulas were changed.

### Post-correction validation

Python 3.12 CI:

```text
89 passed in 0.97s
2016 benchmark observations generated
```

Python 3.11 and Python 3.13 also completed successfully.

The correction materially improved the controlled benchmark, especially `RECOVERY`, so the old `REPORT.md` values were superseded.

## Finding A2 — oracle anchor is a limitation, not hidden leakage

Every benchmark arm currently receives the same explicit known anchor. The anchor is excluded from ranked relevance.

This is visible in the experiment manifest:

```text
anchor_policy = explicit_known_state_excluded_from_ranking
candidate_policy = exhaustive
```

The audit added tests that verify:

- the opaque anchor ID does not appear in natural-language query text;
- relevant node IDs do not appear in query text;
- the anchor is not itself a relevant answer;
- gold node IDs resolve to actual topology nodes.

Conclusion: this is not hidden label leakage, but it makes the experiment an **oracle-anchor retrieval experiment**, not an end-to-end retrieval experiment. Issue #17 is responsible for removing this assumption.

## Finding A3 — retrospective visibility is a limitation

The current benchmark builds the complete trace graph before retrieval. This is valid for retrospective diagnosis/reconstruction tasks but cannot support prospective prediction claims.

Therefore:

- `WHY` is currently retrospective diagnosis;
- `RECOVERY` is historical recovery-path retrieval;
- `WHAT_NEXT` evaluates forward traversal on a completed trace, not real-time prediction.

Issue #30 will implement chronological cutoffs with no future-state visibility.

## Finding A4 — synthetic templates are intentionally favorable to structural evaluation

The gold causal paths are authored from trace templates containing explicit causal edges. This is appropriate for unit-level proof-of-concept testing because ground truth is known exactly, but the result cannot establish external validity.

The audit confirmed that gold labels are authored independently of retriever output. However, the dataset was designed to make causal structure meaningful, so it is not neutral evidence of real-world prevalence or benefit.

Issues #19 and #20 will introduce real and independent public datasets.

## Finding A5 — current dense/lexical arms remain proxy baselines

The current benchmark uses:

```text
Dense proxy   = HashingEmbedder
Lexical proxy = IdfOverlapRetriever
```

The repository already contains stronger adapter implementations, but the published synthetic benchmark does not yet use learned dense retrieval or production BM25.

This is explicitly treated as a limitation. Issue #16 will add competitive BM25 and learned-dense baselines under matched information/context budgets.

## Metric audit

### Recall@K / Precision@K / MRR / nDCG

Hand-calculated fixtures cover:

- partial relevance;
- empty rankings;
- graded relevance;
- duplicate ranking rejection;
- K greater than returned result count.

No correctness defect found.

### Causal Recall@K

Measures retrieval of labeled causal nodes independently from general relevance labels.

A test verifies that a semantically relevant but non-causal node does not count toward causal recall.

No correctness defect found.

### Causal Path Recall

Requires both:

1. all nodes in a gold causal path to be present in the retrieved context plus anchor;
2. every path transition to exist as an explicit `CAUSAL` edge.

A temporal-only path is explicitly tested and scores zero.

No correctness defect found.

### Causal Distance Error

Uses causal BFS in the query-appropriate direction and penalizes missing/disconnected gold evidence by `|V|` rather than silently dropping it.

Wrong-direction retrieval is explicitly tested.

No correctness defect found in the current metric definition. The absolute penalty is dataset-size dependent and should not be compared naïvely across very different graph sizes without normalization; this is a methodological consideration for later datasets.

### Trajectory Reconstruction Accuracy

Reconstructs selected context chronologically and compares it with the labeled trajectory using an LCS-style sequence score.

Tests verify that incorrect temporal ordering and extraneous nodes are penalized.

No correctness defect found. It remains a retrospective reconstruction metric, not a prospective forecasting metric.

### Basin Purity

Measures the fraction of selected nodes belonging to the labeled basin when a basin label is applicable.

No correctness defect found in current fixtures.

### Recovery Path Precision

Measures the fraction of retrieved nodes belonging to labeled recovery nodes when applicable.

No correctness defect found.

### Context-token efficiency

Measures the fraction of deterministic proxy tokens belonging to positively relevant retrieved nodes.

The metric correctly penalizes irrelevant long nodes in existing hand-checkable tests.

No correctness defect found. It is a proxy tokenization metric, not provider-specific LLM billing/token count.

## Adversarial controls added

The audit added at least three independent negative controls:

1. **semantic lookalike control** — a textually similar node connected only temporally must receive zero causal evidence;
2. **temporal-chain placebo** — a temporal path containing all gold nodes cannot satisfy causal-path recall;
3. **wrong-direction control** — descendants retrieved for a `WHY` task do not satisfy ancestor causal ground truth.

Additional leakage-contract assertions ensure opaque node IDs are not exposed through natural-language query text.

## Corrected controlled result

At `K=3`, Full CT-RAG after the direction correction obtains:

```text
Recall@3                 = 0.9583
MRR@3                    = 1.0000
nDCG@3                   = 1.0000
Causal Path Recall       = 0.9167
Causal Distance Error    = 1.5000
Context-token efficiency = 0.8819
```

The earlier published values were:

```text
Recall@3                 = 0.9028
nDCG@3                   = 0.9745
Causal Path Recall       = 0.7500
Causal Distance Error    = 3.5000
Context-token efficiency = 0.8365
```

The updated values are documented in `REPORT.md` together with artifact identity and CI provenance.

## Claims that survived the audit

Within the controlled synthetic/oracle-anchor setting, the following claims remain supported:

- explicit causal/topological structure contains retrieval signal absent from the current semantic/lexical proxies;
- directional causal traversal is important to correctness;
- CT-RAG can recover authored `WHY` ancestor paths at small K;
- CT-RAG can recover authored historical recovery trajectories;
- structural retrieval can increase relevant-context density in these fixtures;
- the benchmark is deterministic and reproducible across the tested Python versions.

## Claims explicitly not established

The audit does not provide evidence that CT-RAG:

- beats learned dense retrieval or production BM25;
- beats GraphRAG or BasinRAG under a matched external protocol;
- works end to end without an oracle anchor;
- generalizes to independent real data;
- predicts future states prospectively;
- identifies interventional/counterfactual effects from observational event logs;
- remains robust under causal metadata corruption;
- improves downstream LLM answer quality;
- scales to production graph sizes.

## Audit disposition

**Internal controlled result: survives with one corrected directional bug.**

The corrected result is stronger than the original result, but the audit does not increase external validity. The next scientific step is to freeze hypotheses/evaluation rules before introducing strong baselines and independent datasets.
