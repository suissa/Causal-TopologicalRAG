# CodeManager evidence robustness v1

This increment adds a reusable ranking layer for CodeManager hypotheses. It extends the commercial experiment's initial artifact crossing with four safeguards:

- evidence is joined only inside the same explicit evidence group;
- producer/consumer/uses-artifact relationships are considered, rather than requiring a literal artifact name in an arbitrary metadata dump;
- repeated samples from one declared independent source count once;
- explicit healthy counter-evidence reduces the ranking score and is returned alongside supporting evidence.

Scores are bounded heuristic priorities, **not probabilities**. The ranker does not consume oracle labels, root-cause IDs, or causal-edge truth. Its output remains an abductive hypothesis.

Run the focused tests with:

```bash
pytest tests/test_codemanager_evidence.py
```

The next integration step is to adapt public projected event metadata into `EvidenceSignal` objects and compare this ranker with the current `codemanager_hypotheses` baseline using leave-one-signal-out and adversarial-group tests. This PR intentionally does not replace the existing baseline until those comparisons run.
