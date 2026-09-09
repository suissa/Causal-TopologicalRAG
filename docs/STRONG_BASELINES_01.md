# Strong Retrieval Baselines — Experiment 01

**Issue:** #16  
**Scientific run:** `34353826174`  
**Commit evaluated:** `ba31889b85588d95074ef16cd506927aded86813`  
**Evaluation split:** `dev` only  
**Final holdout:** sealed

## Purpose

This experiment replaces the original dependency-free semantic/lexical proxies with competitive local retrieval components while preserving exactly the same observable corpus, oracle anchor, K values and exhaustive candidate policy.

It is an internal synthetic validation. It is not external-validity evidence.

## Frozen systems

Lexical baseline:

- `rank-bm25==0.2.2`, `BM25Okapi`;
- `k1=1.5`, `b=0.75`, `epsilon=0.25`.

Learned dense models:

1. `sentence-transformers/all-MiniLM-L6-v2`
   - revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
   - 384 dimensions;
   - normalized embeddings.
2. `sentence-transformers/all-mpnet-base-v2`
   - revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`;
   - 768 dimensions;
   - normalized embeddings.

Hybrid retrieval uses deterministic Reciprocal Rank Fusion with `rank_constant=60`.

## K=3 aggregate result

Across 72 `dev` query instances:

| System | Recall@3 | MRR@3 | nDCG@3 | Causal Path Recall | Causal Distance Error ↓ | Context efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CT-RAG + MPNet + BM25** | **0.9583** | **1.0000** | **0.9851** | **0.9167** | **1.5000** | **0.8819** |
| **Full CT-RAG + MiniLM + BM25** | **0.9583** | **1.0000** | **0.9710** | **0.9167** | **1.5000** | **0.8819** |
| BM25 | 0.0417 | 0.1458 | 0.0281 | 0.0000 | 36.0000 | 0.0420 |
| MiniLM + BM25 hybrid | 0.0382 | 0.1065 | 0.0220 | 0.0000 | 36.0000 | 0.0376 |
| MPNet + BM25 hybrid | 0.0382 | 0.1065 | 0.0220 | 0.0000 | 36.0000 | 0.0376 |
| MiniLM dense | 0.0313 | 0.0556 | 0.0136 | 0.0000 | 36.0000 | 0.0288 |
| MPNet dense | 0.0313 | 0.0556 | 0.0136 | 0.0000 | 36.0000 | 0.0288 |

The learned models do not recover the authored causal trajectory merely by being stronger semantic encoders. In this controlled benchmark, explicit causal/topological structure remains the dominant signal.

This result must not be generalized to ordinary document retrieval. The synthetic task was deliberately authored so semantically similar events can belong to different execution traces.

## Corrected resource cost

The first resource run exposed a warm-cache measurement defect. `SCI-002` records the correction. The following values come only from corrected run `34353826174`, where every learned arm pays for its own cold query embedding while document embeddings remain indexed.

| System | Mean cold query ms |
| --- | ---: |
| BM25 | 1.003 |
| MiniLM dense | 11.494 |
| MiniLM + BM25 hybrid | 11.787 |
| Full CT-RAG + MiniLM/BM25 | 12.652 |
| MPNet dense | 45.578 |
| MPNet + BM25 hybrid | 45.892 |
| Full CT-RAG + MPNet/BM25 | 46.926 |

Relative query-time structural overhead beyond the matched dense+BM25 hybrid is therefore small on these tiny graphs:

- MiniLM path: approximately `0.865 ms/query`;
- MPNet path: approximately `1.034 ms/query`.

These values are not scalability claims; #29 evaluates size-dependent costs.

## Interpretation

Supported on this controlled `dev` split:

- replacing hashing/IDF proxies with learned dense embeddings and BM25 does not remove the causal-topological advantage;
- learned semantic retrieval alone does not reconstruct the authored execution causal paths;
- CT-RAG adds relatively little query-time overhead over the learned encoder for these small graphs.

Not supported by this experiment:

- superiority on natural corpora;
- superiority over all dense or sparse retrievers;
- superiority without a valid anchor;
- scalability claims;
- causal identification beyond the explicit execution relations encoded by the benchmark.

The no-oracle limitation is measured independently in `docs/ANCHOR_DISCOVERY_01.md`.
