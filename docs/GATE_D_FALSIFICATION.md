# Gate D — causal/topological falsification

Status: implemented on generated train/dev mechanisms. The frozen final test was not loaded.

## Protocol

`python -m ctrag.benchmarks.gate_d --output gate-d-results` runs four explicitly exploratory studies at K=3 and seeds 7, 42, and 2024. The command writes a machine-readable result bundle and a SHA-256 manifest tied to the unchanged preregistration fingerprint.

1. Topology placebo controls preserve the exact node IDs, text, embeddings, queries, and K. They remove causal or temporal edge types, reverse random directions, permute endpoints with exact directed-degree-distribution preservation, permute causal targets, and create an endpoint-randomized topology-only sham.
2. Robustness curves use corruption levels 0%, 25%, 50%, 75%, and 100% for missing/removed, noisy, reversed, low-confidence, and mixed-provenance causal edges. Every cell is reported by query mode and with provenance-weighted and unweighted scoring.
3. Evidence contracts separate observational support, interventional evidence, and counterfactual ground truth. Ordinary logs can never exceed `observational_support`. A common-cause confounding control and a labelled simulated binary SCM are represented separately in the artifact.
4. Prospective replay builds a fresh index at each cutoff. Only nodes at or before the anchor timestamp and edges whose two endpoints are visible enter the snapshot. Future labels may be used by the evaluator, never by retrieval.

## Results

The falsification controls do not support a universal “topology always helps” claim.

| Control | WHY Recall@3 | WHAT_NEXT Recall@3 | RECOVERY Recall@3 | Causal-evidence rate (WHY/NEXT/RECOVERY) |
|---|---:|---:|---:|---:|
| intact | 1.000 | 0.875 | 0.667 | 1.000 / 1.000 / 1.000 |
| degree-preserving permutation | 0.042 | 0.521 | 0.556 | 0.083 / 0.083 / 0.083 |
| random direction | 0.229 | 0.635 | 0.667 | 0.417 / 0.333 / 0.833 |
| remove causal | 0.000 | 0.875 | **1.000** | 0 / 0 / 0 |
| topology-only sham | 0.063 | 0.656 | 0.778 | 0.083 / 0.042 / 0.111 |

Meaningful causal evidence collapses when causal topology is removed or randomized. Recall degrades sharply for WHY and usually for WHAT_NEXT. However, causal-edge removal improves RECOVERY Recall@3 from 0.667 to 1.000, and temporal-edge removal changes none of these headline metrics. Those are retained negative/null results, not discarded runs.

The robustness curves identify query-specific failure regions. At 100% deletion, WHY Recall@3 falls from 1.000 to 0 and its causal-evidence rate falls to 0; WHAT_NEXT retains Recall@3 0.875 but loses all causal path evidence; RECOVERY Recall increases while also losing all causal evidence. Fully reversed edges reduce WHAT_NEXT Recall@3 to 0.250 and WHY to 0. Low-confidence edges mainly harm WHY. Random hypothesized edges barely change aggregate recall in this small synthetic corpus, so this experiment does not establish robustness to noise in larger or external graphs.

## Claim contract

`EvidenceLevel` and `validate_causal_claim` enforce the reporting boundary. Historical sequence, association, or divergence is observational. It cannot justify “identified causal effect,” “would have caused,” or “proved causality.” Interventional results require known interventions and labels. Counterfactual claims require paired potential outcomes or ground-truth SCM counterfactual labels.

The current `COUNTERFACTUAL` query mode is therefore an observational divergence retriever. Its result metadata is `observational_support`; its name is retained for API compatibility, not as an identification claim.

## Prospective evaluation

Each prospective row records its cutoff, visible node count, and excluded future-node count. Tests deliberately insert a future node and require the anti-leakage check to fail. Retrospective metrics are never merged into the prospective rows. Terrain/attractor state learned from the full stream is not used.

Prospective “next” evaluation uses only earlier, text-equivalent historical outcomes as retrievable analogues; the true future event remains outside the index. When no earlier analogue exists, the result is a cold-start negative, not silently replaced with retrospective retrieval.

## Limitations and acceptance decision

This Gate D evidence is exploratory because the preregistered final holdout remains sealed. It is a controlled mechanism test, not an external causal-effect study. The full machine-readable curves are in `research/gate-d-v1/gate-d-results.json`; no unfavorable cell is omitted.

Issues #23, #24, #27, and #30 meet their implementation and train/dev acceptance criteria. Gate D can be marked complete at the mechanism-test level while all final confirmatory claims remain pending the later, governed holdout phase.
