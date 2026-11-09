# Paired CodeManager perturbation evaluation

Run the evidence-aware perturbation suite:

```bash
python -m ctrag.benchmarks.codemanager_perturbations
```

Run the paired comparison:

```bash
python -m ctrag.benchmarks.codemanager_paired
```

The paired report uses the same nine deterministic perturbation scenarios and fixed scenario denominator for both arms. It reports candidate coverage, Top-1, Recall@1, Recall@2 and MRR, plus the evidence-aware minus baseline delta.

## Candidate-universe parity

For each scenario, candidate IDs are materialized once as the sorted set of anomalous `code`/`config` signals with an artifact. Both rankers must return a permutation of this exact set; duplicate, missing, or invented candidate IDs fail the evaluation. The report includes the universe and candidate count per scenario. Candidate coverage is reported separately; when gold is absent, Top-1/Recall/MRR remain misses rather than dropping that scenario from the denominator.

## Baseline contract

The comparison arm is a deliberately transparent **legacy-style proxy**: anomalous code/config candidates receive votes for every anomalous observation that mentions the same artifact. It does not enforce evidence-group isolation, temporal ordering, or independent-source deduplication.

This is not represented as the historical production CodeManager implementation. The repository does not currently expose a pinned historical ranker API, so claiming an exact legacy comparison would be misleading. Replacing this proxy with the actual historical implementation requires pinning that revision and its input/output contract.

## Interpretation boundaries

- Synthetic fixtures only; no production incident data or human adjudication.
- Scores are heuristic priorities, not calibrated probabilities.
- Rank changes measure robustness to the defined perturbations, not causal identification or proof that a code artifact caused an incident.
- Cross-group and temporal leakage sensitivity is intentionally visible in the proxy arm and constrained in the evidence-aware arm.
