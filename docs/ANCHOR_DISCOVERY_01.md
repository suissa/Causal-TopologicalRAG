# No-Oracle Anchor Discovery — Experiment 01

**Issue:** #17  
**Scientific run:** `34353826174`  
**Commit evaluated:** `ba31889b85588d95074ef16cd506927aded86813`  
**Evaluation split:** `dev` only  
**Final holdout:** sealed

## Purpose

The original proof-of-concept gave CT-RAG the exact current state as an oracle anchor. This experiment removes that advantage and asks whether the anchor can be discovered from raw query text before causal/topological navigation.

Oracle and discovered-anchor results are intentionally reported separately.

## Anchor selectors

All selectors use the same observable corpus:

- semantic: pinned `all-MiniLM-L6-v2` learned embeddings;
- lexical: `rank-bm25` BM25;
- hybrid: deterministic RRF over semantic + lexical rankings.

No generated opaque event/node ID may appear in the raw query text. The benchmark fails if one is detected.

The primary discovered track chooses one anchor (`Top-1`). A second uncertainty track propagates all three highest-ranked anchor candidates into the causal/topological expansion (`Top-3`).

## Anchor accuracy

### Branching traces

| Selector | Top-1 accuracy | Top-3 contains gold |
| --- | ---: | ---: |
| Semantic | 0.3333 | 0.5833 |
| Lexical | 0.3333 | 0.5833 |
| Hybrid | 0.3333 | 0.5833 |

### Failure/recovery traces

| Selector | Top-1 accuracy | Top-3 contains gold |
| --- | ---: | ---: |
| Semantic | 0.2500 | 0.5833 |
| Lexical | **0.5000** | **0.6667** |
| Hybrid | 0.3889 | 0.6389 |

The experiment therefore rejects any assumption that a gold execution state can currently be recovered reliably from these deliberately ambiguous natural-language queries.

## Downstream K=3 result

Across all 72 `dev` query instances:

| Selector | Track | Recall@3 | MRR@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Context efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid | Oracle | 0.9583 | 1.0000 | 0.9710 | 0.9167 | 1.500 | 0.8819 |
| Hybrid | Discovered Top-1 | 0.4803 | 0.5093 | 0.4507 | 0.4306 | 19.875 | 0.4017 |
| Hybrid | Uncertain Top-3 | 0.5799 | 0.6435 | 0.6015 | 0.4028 | 15.250 | 0.4818 |
| Lexical | Oracle | 0.9583 | 1.0000 | 0.9710 | 0.9167 | 1.500 | 0.8819 |
| Lexical | Discovered Top-1 | **0.5729** | 0.5486 | 0.5454 | **0.5417** | 15.375 | 0.4737 |
| Lexical | Uncertain Top-3 | 0.5729 | **0.6389** | **0.6036** | 0.3750 | 15.375 | 0.4762 |
| Semantic | Oracle | 0.9583 | 1.0000 | 0.9710 | 0.9167 | 1.500 | 0.8819 |
| Semantic | Discovered Top-1 | 0.3484 | 0.4468 | 0.3246 | 0.2917 | 25.500 | 0.3010 |
| Semantic | Uncertain Top-3 | **0.5938** | 0.6435 | 0.5986 | **0.4583** | **14.875** | **0.4918** |

Top-3 uncertainty propagation improves several relevance/ranking measures relative to the same selector's Top-1 path, but it does not restore oracle performance and does not uniformly improve causal path reconstruction.

## Error attribution

For the `discovered_top1` track at K=3, anchor-selection failure accounts for:

- Hybrid: `46 / 72` queries;
- Lexical: `42 / 72` queries;
- Semantic: `51 / 72` queries.

This identifies anchor discovery—not merely causal traversal/reranking—as the main end-to-end failure source in this experiment.

## Ambiguous-query control

The benchmark also creates multi-gold stress queries such as repeated `request accepted` and `operation failed` states without changing the frozen dataset manifest.

Across those controls:

- lexical Top-1 / Top-3 matches any valid anchor: `1.000 / 1.000`;
- hybrid: `1.000 / 1.000`;
- semantic: `0.583 / 0.583`.

This shows that some ambiguity is resolvable at event-type level while exact trace-state disambiguation remains difficult.

## Scientific conclusion

The no-oracle experiment is a **negative/limiting result** relative to the oracle proof-of-concept.

Supported:

- CT-RAG navigation is highly sensitive to anchor correctness;
- explicit propagation of anchor uncertainty can recover part of the loss;
- error attribution can distinguish anchor-selection failure from later retrieval failure.

Not supported:

- the claim that raw natural-language queries currently recover exact execution state reliably;
- pooling oracle and discovered-anchor scores;
- treating the original oracle-anchor benchmark as an end-to-end agent-memory result.

Future anchor work must be tuned only on train/dev. The pristine test split remains sealed.
