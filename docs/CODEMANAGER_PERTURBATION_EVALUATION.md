# CodeManager controlled perturbation evaluation

Run from repository root:

```bash
PYTHONPATH=src python -m ctrag.benchmarks.codemanager_perturbations
pytest tests/test_codemanager_perturbations.py
```

The experiment emits a JSON object with per-scenario ordered candidates, scores,
gold-candidate coverage, Top-1, Recall@1/@2, and MRR. The denominator is the
entire fixed set of scenarios; a missing gold candidate contributes zero to
ranking metrics and coverage rather than disappearing from the denominator.

Perturbations include input reversal, irrelevant evidence, same-source
duplication, independent-source removal, counter-evidence removal, and a
cross-evidence-group leakage attempt. Tests assert determinism and invariance
under duplicate observations and unrelated evidence groups.

## Current boundary

This first executable slice evaluates the evidence-aware ranker only. It does
**not** claim a baseline comparison: the legacy ranker/candidate-generation
contract must be pinned before a paired baseline can be considered fair. The
fixture is synthetic; it is not production validation, human-ground-truth
evaluation, or causal identification. CodeManager scores are heuristic ranking
priorities, not calibrated probabilities.
