# CT-RAG Phase 2 Preregistration

**Status:** preregistered protocol for confirmatory Phase 2 experiments  
**Date frozen:** 2026-09-09  
**Applies after:** completion of internal-validity audit #14

## 1. Purpose

This document freezes the primary scientific questions before CT-RAG is tuned or evaluated against strong learned retrieval baselines and independent datasets.

The goal is not to make CT-RAG win. The goal is to determine whether causal/topological signals provide measurable retrieval value under fair, externally relevant conditions, and to identify the conditions under which they do not.

Changes after this freeze must be classified as either:

- **correctness correction**: fixes a reproducible implementation/metric/data bug and must have a regression test plus a `CHANGELOG_SCIENCE.md` entry;
- **exploratory analysis**: may generate new hypotheses but may not replace or redefine a preregistered primary endpoint;
- **protocol amendment**: must be documented before viewing the affected frozen holdout result and receives a new protocol version/hash.

## 2. Epistemic scope

The preregistered confirmatory claims concern **retrieval over observed execution/event structure**.

The following are not interchangeable:

1. observed execution causality/provenance;
2. inferred causal association;
3. hypothesized causal association;
4. observational trajectory divergence;
5. identified interventional/counterfactual effect.

Ordinary event logs can support 1–4 only to the extent represented by their evidence. They do not by themselves identify item 5.

## 3. Systems to compare

Primary comparison arms, when compatible with a dataset:

1. BM25;
2. learned dense retrieval;
3. learned dense + BM25 hybrid;
4. graph/topology-only retrieval;
5. dense + causal;
6. dense + topological;
7. Full CT-RAG.

Secondary structural comparisons:

8. matched-budget GraphRAG-style baseline;
9. matched BasinRAG reproduction.

The historical `HashingEmbedder` and `IdfOverlapRetriever` remain reproducibility/proxy controls and are not primary competitive baselines.

All arms must receive the same observable corpus, query-time information and context-token budget. A system may not receive gold causal labels or future events unavailable to another system.

## 4. Evaluation tracks

Results must be reported separately for:

### T1 — Oracle-anchor retrospective retrieval

The current controlled track. Used primarily for architectural isolation and regression, not as the headline external-validity result.

### T2 — Discovered-anchor retrospective retrieval

No gold anchor is provided. The system must identify anchor state(s) from the query and observable corpus.

### T3 — Prospective chronological retrieval

For a query at time `t`, only events/nodes/edges observed at or before `t` are indexable. Future topology, future attractors and future terrain reinforcement are forbidden.

### T4 — Independent real/public-data retrieval

Evaluation on traces/datasets not authored by the CT-RAG benchmark generator.

### T5 — Interventional/counterfactual evaluation

Only datasets with known interventions or structural-causal ground truth may contribute to interventional/counterfactual claims.

## 5. Primary hypotheses

### H1 — Diagnostic causal retrieval

For `WHY` tasks with independently available causal/precedence evidence, Full CT-RAG will improve **Causal Path Recall@3** over the strongest non-causal semantic/lexical baseline under the same context budget.

Null H1₀: Full CT-RAG provides no positive paired improvement in Causal Path Recall@3.

Primary endpoint: paired difference in Causal Path Recall@3.

Minimum effect considered practically meaningful: absolute improvement >= **0.05**.

Falsification criterion: if the 95% paired bootstrap confidence interval for the mean improvement includes zero and the point estimate is < 0.05 on the frozen external/independent evaluation, H1 is unsupported.

### H2 — Ranking quality under small context

Full CT-RAG will improve **nDCG@3** over the strongest preregistered BM25/dense/hybrid baseline on tasks where explicit causal/topological evidence is available.

Null H2₀: no positive paired nDCG@3 improvement.

Minimum practically meaningful effect: absolute improvement >= **0.03**.

Falsification criterion: 95% paired bootstrap CI includes zero and point estimate < 0.03 on the frozen external/independent evaluation.

### H3 — Context efficiency

At the same context-token budget, Full CT-RAG will improve relevant-context density over the strongest preregistered semantic/lexical baseline.

Primary endpoint: **context-token efficiency** at the budget corresponding to K=3 in the canonical benchmark and matched token budget in external datasets.

Minimum practically meaningful effect: absolute improvement >= **0.05**.

Falsification criterion: no positive improvement meeting this threshold on the external/independent track.

### H4 — Topology-specific contribution

Destroying meaningful causal/topological structure while holding node text and query set constant will reduce CT-RAG causal/path performance.

Primary endpoint: change in Causal Path Recall@3 under topology-destruction controls.

Minimum expected degradation: absolute drop >= **0.10** for at least the explicit causal-edge permutation/removal control.

Falsification criterion: if destroying causal/topological structure produces no meaningful degradation, the claim that topology causes the observed retrieval gain is unsupported and must be revised.

### H5 — Robustness envelope

CT-RAG will degrade monotonically or near-monotonically as explicit causal metadata is progressively removed/corrupted, and provenance-aware weighting will outperform unweighted treatment of mixed-quality causal edges over at least part of the corruption range.

Primary endpoints:

- Causal Path Recall@3 vs corruption level;
- Causal Distance Error vs corruption level.

This hypothesis is supported only if an interpretable reliability envelope can be identified. Non-monotonic behavior must be investigated, not smoothed away.

### H6 — Dynamic terrain

On chronological repeated-trajectory tasks, terrain-aware retrieval will improve future retrieval for recurrent patterns compared with static CT-RAG without materially harming rare-but-critical path retrieval.

Primary endpoints:

- paired nDCG@3 on recurrent-pattern queries;
- rare-critical-path Recall@3;
- basin drift over time.

A gain on recurrent paths accompanied by a preregistered unacceptable rare-critical-path loss (> 0.05 absolute Recall@3) does not support H6.

## 6. Secondary hypotheses

Secondary analyses include:

- anchor Top-1/Top-k accuracy;
- MRR;
- Recall@K and Precision@K;
- Trajectory Reconstruction Accuracy;
- Basin Purity;
- Recovery Path Precision;
- latency/memory/indexing cost;
- downstream answer correctness and faithfulness.

Secondary endpoints cannot replace a failed primary endpoint.

## 7. Canonical K values and budgets

Canonical ranking K values:

```text
K = 1, 3, 5, 10
```

Primary small-context endpoint:

```text
K = 3
```

For systems whose output units differ materially, comparison must also use matched **context-token budgets**. Token-budget matching takes precedence over raw K when comparing heterogeneous graph/community/document retrieval systems.

No post-hoc K selection may be used for a headline claim.

## 8. Dataset inclusion criteria

A confirmatory dataset must satisfy all of the following applicable conditions:

- source and version are documented;
- license or redistribution/use basis is documented;
- conversion is deterministic and versioned;
- gold targets are original or independently derived, not generated from CT-RAG retrieval output;
- missing/ambiguous causality remains missing/ambiguous;
- all systems receive the same observable information;
- train/dev/test or official split is respected where available.

Synthetic CT-RAG-authored traces remain useful for mechanism tests and negative controls but cannot alone support external-validity headline claims.

## 9. Dataset exclusion criteria

A dataset/query may be excluded only for a predeclared reason such as:

- corrupted/unparseable source record;
- impossible conversion without inventing the target relation;
- licensing restriction;
- exact duplicate crossing a frozen split;
- query has no evaluable target under the declared task.

Exclusions must be machine-recorded with reason codes. Performance-based exclusion is prohibited.

## 10. Holdout policy

The final confirmatory test set must be fingerprinted before tuning.

After holdout freeze:

- retrieval weights, model choices, prompts and thresholds may be tuned only on train/dev;
- final-test labels/aggregates must not be used to choose a configuration;
- a correctness correction discovered before final unblinding may be applied only with a regression test and science changelog entry;
- once final-test aggregate results are viewed, subsequent runs are replication runs, not pristine holdout runs.

## 11. Statistical analysis

Primary comparisons are paired at the finest independent evaluation unit available (normally query within dataset).

Required reporting:

- paired mean/median difference where appropriate;
- 95% paired bootstrap confidence interval;
- effect size;
- raw win/tie/loss counts for ranking metrics where useful;
- exact resampling seed and number of replicates.

Planned bootstrap replicates:

```text
10,000
```

Planned bootstrap seed:

```text
20260909
```

For headline multiple comparisons against several primary baselines, Holm correction will be used when inferential p-values are reported.

P-values must never appear without effect size and confidence interval.

## 12. Primary baseline selection

For each dataset/task family, the headline semantic/lexical comparator is the **best preregistered non-CT baseline on development data**, selected from:

- BM25;
- learned dense model A;
- learned dense model B;
- learned dense + BM25 hybrid.

The final test set may not be used to choose which one is “best.”

GraphRAG and BasinRAG comparisons are reported separately because their retrieval units and indexing assumptions can differ.

## 13. Anchor policy

Oracle-anchor and discovered-anchor results must never be pooled.

Headline end-to-end retrieval claims require discovered anchors or a task where the anchor is legitimately part of the user/system input.

Anchor selection metrics:

- Top-1 accuracy;
- Top-3 accuracy;
- downstream retrieval conditioned on correct vs incorrect anchor.

## 14. Future-state leakage policy

Prospective experiments must enforce a query cutoff timestamp `t`.

Forbidden at query time:

- nodes/events with timestamp > t;
- edges only inferable after t;
- attractors discovered using post-t observations;
- terrain reinforcement/decay updates derived from future transitions;
- labels or terminal outcomes not yet observed.

A failing future-leakage test invalidates the prospective run.

## 15. Negative controls

Confirmatory mechanism testing includes:

- causal-edge removal;
- causal-edge permutation;
- timestamp permutation;
- causation-ID permutation where valid;
- topology removal with node text preserved;
- basin-label permutation;
- placebo/non-causal semantic edges.

If CT-RAG performance is insensitive to destruction of the signal claimed to produce the gain, the causal/topological mechanism claim is unsupported.

## 16. Counterfactual claim policy

Historical divergence retrieval is labeled **observational support**.

A headline counterfactual/interventional claim requires a dataset with known intervention/counterfactual ground truth and is evaluated only in T5.

No ordinary event-log experiment can be promoted to interventional evidence by wording alone.

## 17. Model/parameter tuning policy

Before final holdout evaluation, tuning is allowed on train/dev for:

- learned embedding model selection from the preregistered candidate set;
- BM25 parameters if fixed by dev protocol;
- fusion weights;
- CT-RAG retrieval weights;
- hop budget within the preregistered search range.

All selected values and the selection objective must be persisted.

After test unblinding, changes are exploratory/replication unless correcting a demonstrated correctness defect.

## 18. Stopping rule

The confirmatory phase is not complete merely because one metric is significant.

The primary Phase 2 report requires:

1. H1–H4 evaluated on frozen independent/real/public data where applicable;
2. strong semantic/lexical baselines;
3. no-oracle-anchor track;
4. topology-destruction controls;
5. uncertainty/effect sizes;
6. clean-room replication.

Unsupported hypotheses remain unsupported in the final report.

## 19. Claim language

Allowed before external validation:

> “CT-RAG shows reproducible positive proof-of-concept evidence on controlled synthetic event-sourced traces.”

Allowed after a supported external hypothesis:

> “Under the preregistered datasets/tasks and matched budgets, CT-RAG improved [metric] relative to [baseline] by [effect, CI].”

Not allowed from empirical retrieval experiments alone:

> “CT-RAG is formally proven superior.”

Formal proof, proof-of-concept evidence and empirical generalization are distinct claim types.
