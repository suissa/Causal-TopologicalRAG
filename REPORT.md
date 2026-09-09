# CT-RAG Experimental Validation Report

**Project:** Causal-Topological RAG (CT-RAG)  
**Validated commit:** `f4827e223802b1c10cd7d5c268b00da561526882`  
**CI run:** https://github.com/suissa/Causal-TopologicalRAG/actions/runs/34312104975  
**Date:** 2026-09-09

## Executive conclusion

The current experiment provides **positive empirical evidence that the Causal-Topological RAG concept works for the controlled problem it was designed to test**: retrospective retrieval over event-sourced execution histories where explicit causal relations, topology, distractors, failures, recovery paths, and branching outcomes are known.

The strongest result is not merely that CT-RAG eventually retrieves the same evidence with a larger context window. At small retrieval budgets it finds the correct causal evidence **earlier, in better order, with less irrelevant context, and reconstructs causal paths substantially better than semantic or lexical retrieval alone**.

At `K=3`, across all 72 query instances:

| System | Recall@3 | MRR@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Context-token efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CT-RAG** | **0.9028** | **1.0000** | **0.9745** | **0.7500** | **3.5000** | **0.8365** |
| Dense + Causal | 0.5509 | 0.7870 | 0.7171 | 0.2500 | 16.1667 | 0.4821 |
| Graph / Topology only | 0.7500 | 0.8588 | 0.6331 | 0.4792 | 13.8333 | 0.7230 |
| Dense + Topological | 0.2836 | 0.3171 | 0.2064 | 0.0833 | 30.0833 | 0.2944 |
| Dense only | 0.0428 | 0.1273 | 0.0292 | 0.0000 | 35.8333 | 0.0455 |
| Dense + Lexical | 0.0382 | 0.1319 | 0.0255 | 0.0000 | 36.0000 | 0.0388 |
| Lexical only | 0.0278 | 0.0787 | 0.0161 | 0.0000 | 36.0000 | 0.0256 |

Relative to graph/topology-only at `K=3`, full CT-RAG improves:

- Recall by **20.4%**;
- nDCG by **53.9%**;
- Causal Path Recall by **56.5%**;
- context-token efficiency by **15.7%**;
- causal distance error is reduced by **74.7%**.

Relative to Dense + Causal, full CT-RAG improves nDCG by **35.9%**, Causal Path Recall by **200%**, and reduces causal distance error by **78.4%**.

This supports the central CT-RAG hypothesis for these experiments:

> Retrieval quality over execution history improves when semantic evidence is combined with explicit causal and topological structure rather than treating memories as independent nearest-neighbor items.

It does **not** yet prove that CT-RAG is superior on arbitrary real-world corpora or with production embedding models. The present result is a controlled proof-of-concept, not a claim of universal external validity.

---

## 1. What was executed

The GitHub Actions CI executed the complete suite on Python **3.11, 3.12, and 3.13**. Every job completed successfully.

For Python 3.12, the CI log records:

```text
25 passed in 0.45s
Wrote 2016 query/K/baseline observations to benchmark-results
```

The benchmark produced six reproducibility artifacts for every Python version:

```text
config.json
datasets.json
results.json
results.csv
summary.csv
table.csv
```

The 2,016 observations are exactly:

```text
72 query instances
× 7 retrieval arms
× 4 K values (1, 3, 5, 10)
= 2,016 observations
```

The 72 query instances contain:

- 36 `WHY` queries;
- 24 `WHAT_NEXT` queries;
- 12 `RECOVERY` queries.

They are generated over deterministic failure/recovery and branching execution traces using three seeds.

---

## 2. Controlled retrieval arms

Seven ablations were evaluated against exactly the same candidate corpus and known anchor:

1. `lexical_only`
2. `dense_only`
3. `dense_lexical`
4. `graph_topology_only`
5. `dense_causal`
6. `dense_topological`
7. `full_ctrag`

The benchmark uses exhaustive candidates, which is important: an ablation is not advantaged or disadvantaged by an earlier candidate-pruning stage.

The current "dense" signal is the deterministic `HashingEmbedder` proxy already implemented in the project. It is intentionally local and reproducible and is **not** a learned Sentence Transformer/OpenAI embedding benchmark. Likewise, the lexical baseline is the project's normalized token/IDF overlap, not production BM25.

This means the experiment isolates the architectural value of causal/topological signals, but a later experiment is still required against strong learned dense and BM25 baselines.

---

## 3. Primary result: CT-RAG dominates at small context budgets

The most useful operating point is `K=3`, because this tests whether the architecture can recover causal context without simply retrieving most of the graph.

### 3.1 Ranking quality

Full CT-RAG obtains:

```text
MRR@3  = 1.0000
nDCG@3 = 0.9745
```

The nearest alternatives are:

```text
Dense + Causal       nDCG@3 = 0.7171
Graph/Topology only  nDCG@3 = 0.6331
```

Thus CT-RAG is not merely retrieving relevant nodes: it is placing the important nodes at the top of the context.

### 3.2 Causal reconstruction

At `K=3`:

```text
Full CT-RAG             Causal Path Recall = 0.7500
Graph/Topology only     Causal Path Recall = 0.4792
Dense + Causal          Causal Path Recall = 0.2500
Dense only              Causal Path Recall = 0.0000
Lexical only            Causal Path Recall = 0.0000
```

The same pattern appears in causal distance error:

```text
Full CT-RAG               3.50
Graph/Topology only      13.83
Dense + Causal           16.17
Dense only               35.83
Lexical only             36.00
```

Lower is better. Full CT-RAG reduces the graph/topology-only error by 74.7%.

This is direct evidence for the part of CT-RAG that is different from ordinary RAG: **the retrieved context preserves substantially more of the causal trajectory connecting the anchor to its relevant evidence**.

---

## 4. WHY queries: the strongest validation of the hypothesis

`WHY` is the most direct test of causal retrieval because the system must recover causal ancestors of an observed state or failure.

At `K=3`:

| System | Recall@3 | MRR@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Trajectory reconstruction | Token efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CT-RAG** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **0.9167** | **0.8750** |
| Graph/Topology only | 0.8750 | 0.8565 | 0.7145 | 0.7500 | 6.5000 | 0.8542 | 0.7768 |
| Dense + Causal | 0.3426 | 0.6574 | 0.5467 | 0.0000 | 23.6667 | 0.4653 | 0.1978 |
| Dense only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 36.0000 | 0.2500 | 0.0000 |
| Lexical only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 36.0000 | 0.2500 | 0.0000 |

This is the clearest evidence in the current experiment.

For `WHY` queries, full CT-RAG reconstructs every gold causal path at `K=3`, while the dense and lexical baselines fail completely on the authored causal evidence.

That outcome is consistent with the motivation behind CT-RAG: semantic resemblance alone does not identify **what caused the current state**.

---

## 5. RECOVERY queries

Recovery tasks test whether the retriever can recover the historical remediation path after a failure.

At `K=3`:

| System | Recall@3 | nDCG@3 | Recovery Path Precision | Causal Distance Error ↓ |
| --- | ---: | ---: | ---: | ---: |
| **Full CT-RAG** | **0.6667** | **0.8473** | **0.6667** | **12.0** |
| Graph/Topology only | 0.5833 | 0.5816 | 0.5833 | 23.0 |
| Dense + Causal | 0.5278 | 0.6628 | 0.5278 | 17.0 |
| Dense only | 0.0278 | 0.0303 | 0.0278 | 35.0 |
| Lexical only | 0.0000 | 0.0000 | 0.0000 | 36.0 |

At `K=5`, full CT-RAG reaches:

```text
Recall                    = 1.0000
Causal Path Recall        = 1.0000
Causal Distance Error     = 0.0000
MRR                       = 1.0000
nDCG                      = 0.9409
Basin Purity              = 1.0000
```

Graph/topology-only also reaches full recall/path reconstruction at K=5, but its nDCG is only `0.7517`. In other words, topology can eventually recover the same path, but CT-RAG ranks the recovery evidence substantially better.

---

## 6. WHAT_NEXT queries

At `K=3`, full CT-RAG and Dense + Causal tie on the main relevance metrics:

```text
Recall@3             = 0.8750
MRR@3                = 1.0000
nDCG@3               = 1.0000
Causal Path Recall   = 0.7500
```

However, CT-RAG obtains basin purity `1.0000`, while Dense + Causal obtains `0.8611`.

At `K=5`, both reach complete relevant/causal path recall, while CT-RAG preserves perfect basin purity.

This is an important nuance: **not every task requires every CT-RAG component to win**. For forward traversal over these synthetic traces, explicit causal edges are already a very strong signal. The topological component contributes primarily by keeping the retrieval within the correct behavioral region.

---

## 7. Paired dominance analysis

Because every retrieval arm is evaluated on exactly the same query instances, paired comparisons are possible.

At `K=3`, for nDCG:

### Full CT-RAG vs Dense only

```text
Wins:   72
Ties:    0
Losses:  0
```

Exact two-sided sign test: `p ≈ 4.24 × 10^-22`.

### Full CT-RAG vs Graph/Topology only

```text
Wins:   65
Ties:    6
Losses:  1
```

Exact two-sided sign test excluding ties: `p ≈ 1.82 × 10^-18`.

### Full CT-RAG vs Dense + Causal

```text
Wins:   44
Ties:   28
Losses:  0
```

Exact two-sided sign test excluding ties: `p ≈ 1.14 × 10^-13`.

For Recall@3 versus Graph/Topology only:

```text
Wins:   26
Ties:   46
Losses:  0
p ≈ 2.98 × 10^-8
```

For Causal Path Recall versus Graph/Topology only:

```text
Wins:   22
Ties:   50
Losses:  0
p ≈ 4.77 × 10^-7
```

For Causal Distance Error versus Graph/Topology only:

```text
Better: 33
Ties:   39
Worse:   0
p ≈ 2.33 × 10^-10
```

These tests should be interpreted within the synthetic benchmark design; the seeds vary terminology, identifiers and insertion order over fixed trace templates and are not independent real-world datasets. Still, they show that the observed aggregate difference is not produced by one isolated query.

---

## 8. What happens when K increases

At `K=5`, full CT-RAG reaches:

```text
Recall                    = 1.0000
Causal Recall             = 1.0000
Causal Path Recall        = 1.0000
Causal Distance Error     = 0.0000
MRR                       = 1.0000
nDCG                      = 0.9902
```

Graph/topology-only also reaches complete recall and causal path recovery at K=5, but:

```text
MRR  = 0.8588
nDCG = 0.7473
```

This distinction matters. With enough slots, topology alone can include the required nodes, but CT-RAG puts the correct evidence earlier.

At `K=10`, many structural systems converge in recall because a large fraction of the small synthetic graph is being retrieved. Token efficiency correspondingly falls. This confirms why small-K evaluation is necessary: a retriever that simply returns most of the graph can hide ranking defects.

---

## 9. Context efficiency

At K=3:

```text
Full CT-RAG            0.8365
Graph/Topology only    0.7230
Dense + Causal         0.4821
Dense only             0.0455
Lexical only           0.0256
```

The metric is the fraction of deterministic proxy tokens belonging to positively relevant retrieved nodes.

Full CT-RAG therefore places substantially less irrelevant material into the small context window than semantic/lexical baselines in this benchmark.

This supports a practical argument for causal-topological retrieval: better structural retrieval can improve not only evidence coverage but also the **information density of the context sent downstream**.

---

## 10. Cross-version reproducibility

The CI generated independent benchmark artifacts on Python 3.11, 3.12 and 3.13.

Observed reproducibility:

- `datasets.json`: byte-identical on all three Python versions;
- `table.csv`: byte-identical on all three Python versions;
- Python 3.12 and 3.13 `results.csv` and `summary.csv`: byte-identical;
- Python 3.11 differs from 3.12/3.13 in only five nDCG floating-point values, with maximum absolute difference `1.1102230246251565e-16`;
- all rankings and substantive metrics are therefore numerically equivalent to machine precision.

The `config.json` files intentionally differ because they persist the Python runtime version.

This is sufficient to regard the experiment as reproducible across the tested interpreter versions for scientific purposes.

---

## 11. What this experiment proves

Within the controlled trace benchmark, the data supports the following claims.

### Supported

1. Explicit causal/topological information contains retrieval signal not recoverable from the current semantic/lexical proxy alone.
2. Combining causal, topological, semantic and other retrieval components can outperform individual components.
3. CT-RAG can retrieve causal ancestors for diagnostic `WHY` queries with much smaller context than semantic retrieval.
4. CT-RAG can reconstruct historical failure/recovery trajectories.
5. Basin/topological confinement can reduce retrieval leakage into unrelated regions.
6. CT-RAG can achieve higher relevant-token density in the retrieved context.
7. The implementation and benchmark are deterministic/reproducible across Python 3.11–3.13 to floating-point precision.

### Not yet supported

The experiment does **not** establish that:

1. CT-RAG beats state-of-the-art learned embedding systems on real data;
2. CT-RAG beats production BM25 implementations;
3. CT-RAG beats BasinRAG or GraphRAG on their native benchmark suites;
4. explicit event causation fields always represent true philosophical/statistical causality;
5. the current weights generalize without calibration;
6. the observed gains survive noisy, missing, incorrect, or inferred causal edges;
7. CT-RAG improves final LLM answer accuracy on a real downstream task;
8. the system identifies unseen counterfactual effects.

Those claims require separate experiments.

---

## 12. Why this is still a meaningful validation

The benchmark is synthetic by design, but it is not a tautological "CT-RAG retrieves what CT-RAG generated" test.

Ground truth is authored independently from retrieval output. The corpus includes semantically similar distractors, temporally adjacent non-causes, branching outcomes, failures, recovery sequences and explicit causal edges. Every baseline sees the same candidates and anchor.

The key falsifiable question is:

> Given the same memory corpus, does preserving and exploiting causal/topological structure improve retrieval of the execution evidence that actually belongs to the target trajectory?

For this controlled experiment, the answer is **yes**.

The semantic and lexical arms perform poorly despite seeing the same text. Adding causal information helps. Adding topology helps. Combining the signals in full CT-RAG produces the strongest overall small-K result.

That is exactly the architectural proposition the first experiment was designed to test.

---

## 13. Verdict

**Status: concept validated at proof-of-concept level.**

The current data is sufficient to reject the weaker null architectural assumption that semantic/lexical similarity alone is equally effective for retrieving causal execution history in these controlled scenarios.

The result that matters most is:

```text
WHY queries @ K=3

Full CT-RAG:
Recall              1.0000
MRR                 1.0000
nDCG                1.0000
Causal Path Recall  1.0000
Causal Dist. Error  0.0000

Dense only:
Recall              0.0000
MRR                 0.0000
nDCG                0.0000
Causal Path Recall  0.0000
Causal Dist. Error 36.0000
```

The next scientific milestone should therefore no longer ask only "does the concept work at all?". The controlled benchmark says it does.

The next question is:

> **How much of this gain survives when CT-RAG is evaluated on real event-sourced traces, learned dense embeddings, BM25, incomplete causal metadata, and strong GraphRAG/BasinRAG baselines?**

That is the experiment required to move from proof-of-concept validation to external empirical validation.
