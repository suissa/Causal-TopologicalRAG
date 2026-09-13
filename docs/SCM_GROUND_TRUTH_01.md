# Independent SCM ground-truth experiment 01

Status: exploratory train/dev experiment for #38 and #41. The final CT-RAG holdout was not loaded.

## Question and information boundary

This experiment separates three questions that must not be conflated:

1. Can a retriever recover the correct logged `do(X=0)`/`do(X=1)` evidence when a true causal link is supplied?
2. Can observational data identify an average treatment effect under the required assumptions?
3. Can CT-RAG itself estimate an individual counterfactual outcome?

The generator creates labels before retrieval from fixed structural equations with true ATE and individual effect 2.0. Three families are evaluated across five seeds: an unconfounded linear chain, an observed-confounder model, and a hidden-confounder negative control. Each has observational and randomized interventional records.

Indexable fields are restricted to family, split, observed treatment/covariates/outcome and intervention label. Exogenous noise, hidden confounders, paired counterfactual outcomes and true individual effects are blocked by a tested leakage guard.

## Results

Macro means across five seeds:

| SCM family | Dense intervention-pair Recall@2 | CT-RAG with supplied true graph | Observational ATE error | Adjusted ATE error | Randomized-do ATE error |
|---|---:|---:|---:|---:|---:|
| linear chain | 0.000 | **1.000** | 0.022 | 0.022 | 0.021 |
| observed confounder | 0.000 | **1.000** | 1.806 | **0.036** | 0.330 |
| hidden confounder | 0.000 | **1.000** | 1.806 | 1.806 | 0.330 |

The retrieval result is strong and narrow: supplied causal topology reliably navigates from an observational episode to its correct interventional evidence, while semantic similarity alone does not identify the paired unit.

The causal-estimation result preserves the expected negative control. Naive observational estimation is badly biased under confounding. Adjustment succeeds only when the confounder is observed; it cannot repair the hidden-confounder family. Randomized intervention estimates are substantially less biased, with finite-sample error.

## Negative result and claim decision

Current CT-RAG does not estimate an SCM or the exogenous state required for individual counterfactual identification. Counterfactual-estimation coverage is therefore reported as `0.0`, not imputed from retrieved outcomes.

Consequently:

- supported: CT-RAG can retrieve labelled interventional evidence when valid causal topology is supplied;
- not supported: CT-RAG discovers that topology from observational samples;
- not supported: CT-RAG identifies individual counterfactual outcomes;
- not supported: retrieval of supplied edges is causal discovery.

The full 15 family/seed cells and information contract are in `research/scm-ground-truth-v1/scm-results.json`.
