# CT-RAG Scientific Validation and Falsification Plan

## Objective

Move CT-RAG from a controlled proof-of-concept to an empirically defensible research result.

The goal of this phase is **not** to force CT-RAG to outperform every baseline. The goal is to determine, with reproducible evidence:

1. whether causal/topological signals add retrieval value beyond semantic/lexical retrieval;
2. which tasks benefit and which do not;
3. how much of the gain depends on oracle anchors, clean causal metadata, synthetic construction or future visibility;
4. how robust the method is to missing/noisy topology and concept drift;
5. whether improved retrieval translates into better downstream answers;
6. the computational cost of those gains;
7. the conditions under which the current CT-RAG formulation must be revised or rejected.

## Scientific rules

- Negative and null results remain first-class outputs.
- Temporal order never becomes causality by assumption.
- Observed, inferred and hypothesized causal edges remain distinct.
- Observational divergence is not presented as an identified counterfactual effect.
- Any logic correction discovered during experiments receives a minimal failing regression test first.
- Correctness bugs may be repaired on train/dev; the frozen final holdout is not used for iterative tuning.
- No headline claim may be based only on the authored synthetic generator.
- All headline comparisons use matched information/context budgets where protocol-compatible.
- Every headline number must trace to machine-readable raw output, configuration and code/data fingerprints.

## Execution gates

### Gate A — Internal validity and experiment freeze

Issues:
- #14 benchmark/metric/leakage audit;
- #15 preregistration;
- #18 held-out train/dev/test protocol;
- #33 mandatory scientific logic-correction gate.

Exit condition: the benchmark and primary hypotheses are frozen before strong-baseline tuning or final-test exposure.

### Gate B — Competitive retrieval baselines

Issues:
- #16 strong BM25 + learned dense baselines;
- #17 non-oracle anchor discovery;
- #21 matched GraphRAG comparison;
- #22 matched BasinRAG reproduction/comparison.

Exit condition: CT-RAG is compared with competitive alternatives without privileged corpus, model, anchor or token access.

### Gate C — External validity

Issues:
- #19 real event-sourced traces;
- #20 independent public datasets;
- #31 blinded human annotation/inter-annotator agreement.

Exit condition: at least one headline analysis depends on data not authored by the CT-RAG synthetic generator.

### Gate D — Falsification of the causal/topological mechanism

Issues:
- #23 topology destruction/permutation/placebo controls;
- #24 causal-edge corruption robustness;
- #27 observational vs interventional/counterfactual separation;
- #30 prospective chronological evaluation.

Exit condition: we know whether measured gains actually disappear when meaningful causal/topological structure is removed or corrupted, and whether the method survives realistic chronological constraints.

### Gate E — Dynamic terrain, downstream utility and systems evidence

Issues:
- #26 longitudinal terrain/concept-drift validation;
- #28 downstream answer quality/faithfulness;
- #29 scalability/resource cost.

Exit condition: retrieval improvements, if present, are connected to agent utility and bounded by their operational cost/failure modes.

### Gate F — Statistical inference and replication

Issues:
- #25 confidence intervals/effect sizes/paired tests;
- #32 clean-room replication;
- #34 final scientific validation report and claim matrix.

Exit condition: every preregistered primary hypothesis is classified as supported, unsupported or inconclusive with effect size and uncertainty, and headline results reproduce from a clean environment.

## Core comparisons

At minimum, the final comparison matrix should contain:

- lexical proxy (historical IDF overlap);
- dense proxy (historical HashingEmbedder);
- strong BM25;
- learned dense retrieval;
- learned dense + BM25 hybrid;
- graph/topology-only;
- dense + causal;
- dense + topological;
- full CT-RAG;
- terrain-aware CT-RAG where applicable;
- matched GraphRAG;
- matched BasinRAG where protocol-compatible.

## Primary task families

- semantic/relevance retrieval;
- `WHY` causal ancestor retrieval;
- `WHAT_NEXT` prospective/retrospective transition retrieval;
- `RECOVERY` historical recovery-path retrieval;
- trajectory reconstruction;
- basin/attractor localization;
- downstream diagnostic answer quality;
- observational divergence;
- interventional/counterfactual evaluation only on datasets with appropriate ground truth.

## Required controls

- topology removed;
- causal edges permuted;
- timestamps permuted;
- causation IDs permuted;
- basin membership permuted;
- false causal edges injected;
- true causal edges dropped;
- semantic lookalikes with no causal relation;
- causally adjacent events with low semantic similarity;
- future-state visibility prohibited in prospective tracks.

## Statistical reporting

Headline comparisons must include:

- paired effect size;
- confidence interval;
- paired resampling/randomization test where appropriate;
- multiplicity policy for primary comparisons;
- raw per-query/per-seed/per-dataset observations;
- explicit handling of missing/non-applicable metrics.

Point estimates alone are insufficient for final scientific claims.

## Logic-correction policy

If an experiment produces an unexpected result:

1. reproduce it;
2. determine whether it is a correctness bug, modeling assumption, dataset artifact or genuine scientific result;
3. create a minimal failing test before fixing a correctness bug;
4. record the correction in `CHANGELOG_SCIENCE.md`;
5. rerun train/dev analyses;
6. do not use final-holdout feedback to tune the fix;
7. preserve both pre- and post-correction results when they affected a previously reported claim.

A result where CT-RAG loses is **not** itself evidence of a bug.

## Final deliverable

`SCIENTIFIC_VALIDATION_REPORT.md` must contain a claim matrix:

```text
claim
  -> preregistered hypothesis
  -> dataset/split
  -> comparison baseline
  -> metric
  -> effect size
  -> confidence interval/statistical test
  -> status: supported | unsupported | inconclusive
  -> artifact/hash
```

It must separate:

- synthetic proof-of-concept evidence;
- real event-trace evidence;
- independent public-dataset evidence;
- prospective evidence;
- interventional/counterfactual evidence;
- scalability evidence.

## Definition of scientific-validation done

This phase is complete only when issue #35 can be closed with all required gates satisfied and the final report makes no claim stronger than the evidence supports.
