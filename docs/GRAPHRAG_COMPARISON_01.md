# Matched GraphRAG-style Comparison — Experiment 01

**Issue:** #21  
**Scientific run:** `34353826174`  
**Commit evaluated:** `ba31889b85588d95074ef16cd506927aded86813`  
**Evaluation split:** `dev` only  
**Final holdout:** sealed  
**Acceptance status:** complete

## Scope

This experiment compares CT-RAG against a reproducible graph-based retrieval baseline under matched learned embeddings, lexical retrieval, corpus, oracle anchor, event-node retrieval units and K values.

The comparator is explicitly a **GraphRAG-style semantic relation graph**, not the Microsoft GraphRAG implementation and not an LLM entity/community extraction system.

## Information boundary

The semantic GraphRAG-style baseline may use:

- event text;
- the same learned sentence embedding model as CT-RAG;
- the same BM25 lexical component;
- the same oracle anchor;
- an undirected semantic k-NN relation graph (`k=3`);
- graph traversal up to 3 hops;
- deterministic RRF fusion of dense, lexical and relation-graph rankings.

It may not use:

- CT-RAG causal edges;
- causation IDs;
- execution topology;
- gold relevance labels.

A regression test adds an arbitrary causal edge to an otherwise identical dataset and verifies that the semantic relation graph remains unchanged.

## K=3 aggregate result

Across all 72 `dev` query instances:

| Model | System | Recall@3 | MRR@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Context efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM | **Full CT-RAG** | **0.9583** | **1.0000** | **0.9710** | **0.9167** | **1.500** | **0.8819** |
| MiniLM | Semantic GraphRAG-style | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 36.000 | 0.0000 |
| MPNet | **Full CT-RAG** | **0.9583** | **1.0000** | **0.9851** | **0.9167** | **1.500** | **0.8819** |
| MPNet | Semantic GraphRAG-style | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 36.000 | 0.0000 |

The zero score of the semantic graph comparator is not interpreted as a general GraphRAG failure. It is diagnostic of this controlled benchmark: repeated semantically similar event descriptions occur across distinct execution traces, while the gold evidence is trace-specific. Semantic relations therefore preferentially connect textually similar states without identifying which execution actually caused the target state.

That property is exactly why this benchmark is useful for mechanism isolation, but it also makes it unsuitable for a broad claim that CT-RAG is superior to GraphRAG on natural corpora.

## Per-task result

The same qualitative outcome holds for `WHY`, `WHAT_NEXT` and `RECOVERY` in this synthetic split. Full CT-RAG retains causal path evidence; the semantic relation graph does not recover the authored trace-specific gold nodes at K=3.

For Full CT-RAG with MPNet:

| Mode | Recall@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Context efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| WHY | 1.000 | 1.000 | 1.000 | 0.0 | 0.8750 |
| WHAT_NEXT | 0.875 | 0.9553 | 0.750 | 4.5 | 0.8333 |
| RECOVERY | 1.000 | 1.000 | 1.000 | 0.0 | 1.0000 |

## Resource measurements

Mean cold-query latency:

| Model | Semantic GraphRAG-style | Full CT-RAG |
| --- | ---: | ---: |
| MiniLM | 11.658 ms | 12.438 ms |
| MPNet | 44.578 ms | 45.340 ms |

Semantic relation-graph indexing per generated dataset/seed averaged:

- MiniLM: `108.2 ms`;
- MPNet: `301.2 ms`.

These are tiny synthetic graphs; they are not scaling claims.

## Interpretation

Supported for this controlled mechanism test:

- semantic graph connectivity is not interchangeable with observed execution causality;
- giving semantic graph retrieval the same learned dense and lexical components does not recover trace-specific causal evidence here;
- CT-RAG's gain cannot be attributed merely to using a graph data structure.

Not supported:

- a claim against Microsoft GraphRAG;
- a claim against entity/relation GraphRAG on natural corpora;
- a universal claim that semantic graph traversal is ineffective;
- an external-validity claim.

Issue #22 separately reproduces the actual upstream BasinRAG implementation so the topological comparison is not reduced to an in-house surrogate.
