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

The commercial experiment now adapts public projected event metadata into `EvidenceSignal` objects and keeps the existing `codemanager_hypotheses` baseline for side-by-side evaluation. The next evaluation step is to report per-incident CodeManager recall and compare against that baseline on held-out incidents.


## Experiment integration update

The commercial experiment now adapts public projected observations into
`EvidenceSignal` values and emits `codemanager_evidence_ranker` alongside
the legacy `codemanager_hypotheses` output. The specialist evaluation is scoped
to the current incident and is counted only when the candidate set contains
that incident's held-out root-cause event; unrelated incident types are not
counted as CodeManager false negatives.

The focused tests now include leave-one-independent-signal-out behavior and
evidence-group isolation. They have been added to the PR, but have not yet been
executed in a Python environment. Do not treat the implementation as validated
until CI or a local test run passes.


## Artifact-link coverage follow-up

The pricing-drift fixture now declares explicit artifact relationships among the
configuration candidate, conversion metric, checkout trace, logs, and symptom.
The end-to-end test asserts that the configuration candidate is ranked using
all four signal kinds. This closes an important fixture-quality gap: an evidence
group alone must never be sufficient to join observations when the artifacts do
not connect. The follow-up commit must pass CI before these new assertions are
considered validated.


## Paired baseline comparison

The evaluation now emits a paired row for each CodeManager method on every
scenario under the separate task `codemanager_diagnosis`. Both methods receive
the same incident-scoped candidate universe. Each row reports:

- `candidate_count`: number of hypotheses generated for that incident;
- `gold_in_candidates`: whether the held-out root cause was generated at all;
- `rank`, `top1`, `recall_at_k`, and `mrr`: ranking metrics, with missing
  causes counted as misses rather than silently dropped from the denominator.

This separates candidate-generation coverage from ranking quality and avoids
inflating the ranker's score by evaluating only incidents where it happened to
generate the correct candidate. The legacy baseline and evidence-aware ranker
are summarized separately in `results.json`. These remain synthetic,
scenario-dependent results; they are not evidence of external validity or
calibrated causal probabilities. CI validation is required before interpreting
the resulting numbers.
