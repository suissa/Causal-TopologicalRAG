# External Validity Report 01 — Public GitHub Actions Traces

Run: `34727234749`  
Artifact: `external-validity-python-3.12`  
Artifact SHA-256: `b54b3cf27997497be4e37182e5f1e6fafae800ccb58111a5c33f62273a39fd9c`  
Final preregistered holdout: **sealed / not executed**

## Scope

This experiment evaluates real public GitHub Actions execution histories that were not authored by the synthetic benchmark generator. Step order is represented as `TEMPORAL` topology only. No causal edge is inferred from adjacency.

The CT-RAG arm in this report uses the observed current step as an oracle anchor. Therefore the result isolates navigation over real procedural topology and is not an end-to-end anchor-discovery score.

## Results at K=3

| Dataset | Repository | Queries | Arm | Recall@3 | MRR | nDCG@3 | Trajectory reconstruction |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| ctrag-science-success | suissa/Causal-TopologicalRAG | 19 | Dense | 0.4211 | 0.3596 | 0.3753 | 0.3553 |
|  |  |  | Hybrid | 0.3684 | 0.3333 | 0.3421 | 0.3421 |
|  |  |  | Lexical | 0.3158 | 0.3158 | 0.3158 | 0.3289 |
|  |  |  | **CT-RAG temporal/topological** | **1.0000** | **0.8947** | **0.9223** | **0.5000** |
| ctrag-science-failure | suissa/Causal-TopologicalRAG | 20 | Dense | 0.4000 | 0.3417 | 0.3565 | 0.3500 |
|  |  |  | Hybrid | 0.3500 | 0.3167 | 0.3250 | 0.3375 |
|  |  |  | Lexical | 0.3000 | 0.3000 | 0.3000 | 0.3250 |
|  |  |  | **CT-RAG temporal/topological** | **1.0000** | **0.9000** | **0.9262** | **0.5000** |
| basinrag-ci-failure | Basinfy/BasinRAG | 8 | Dense | 0.3750 | 0.1667 | 0.2202 | 0.3438 |
|  |  |  | Hybrid | 0.3750 | 0.1667 | 0.2202 | 0.3438 |
|  |  |  | Lexical | 0.3750 | 0.1667 | 0.2202 | 0.3438 |
|  |  |  | **CT-RAG temporal/topological** | **1.0000** | **0.6875** | **0.7693** | **0.5000** |
| requests-lock-success | psf/requests | 2 | Dense | **1.0000** | 0.5000 | 0.6309 | **0.6667** |
|  |  |  | Hybrid | **1.0000** | 0.5000 | 0.6309 | **0.6667** |
|  |  |  | Lexical | **1.0000** | 0.5000 | 0.6309 | **0.6667** |
|  |  |  | CT-RAG temporal/topological | **1.0000** | **0.7500** | **0.8155** | **0.6667** |

## What this supports

On these real procedural traces, when the current execution state is known, explicit temporal/topological navigation consistently places the observed next step within K=3 and usually ranks it substantially above semantic/lexical retrieval.

The independent `psf/requests` trace is an important control: dense, lexical and hybrid already achieve Recall@3 = 1.0, so CT-RAG has no recall advantage there. Its gain is only ranking quality (MRR/nDCG). This prevents the external benchmark from being interpreted as a universal semantic-baseline failure.

## What this does not support

This experiment contains **zero causal edges**. Consequently it provides no evidence that temporal adjacency is causal, no causal-path claim, and no causal-recall result. The failure-oriented `WHY` and `RECOVERY` labels in this dataset mean observed predecessor / observed post-failure continuation only.

The CT-RAG arm uses oracle anchors. Gate B already established that anchor discovery is a major end-to-end bottleneck; this external experiment does not remove that limitation.

The `psf/requests` dataset has only two evaluable next-step queries and must not receive the same evidential weight as a large benchmark.

## Conclusion

The first external evidence supports a narrow claim: **topological navigation over a known state can improve ranking of observed procedural successors on real execution traces, even when no causal metadata exists.**

It does not yet validate causal retrieval on real traces. That requires a source containing explicit, provenance-bearing dependency/causation relations rather than mere sequence.
