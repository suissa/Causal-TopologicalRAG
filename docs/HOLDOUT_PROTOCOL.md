# Phase 2 Holdout Protocol

This document defines how CT-RAG separates development from confirmatory evaluation after the Phase 2 preregistration.

## Frozen split specification

The split specification is committed at:

```text
research/holdout/manifest-v1.json
```

Its SHA-256 is:

```text
f0e3def445bd5c4d8c3cbc9c66e64f3a752c7bc782d15a8332bede15671e5137
```

The split seeds are disjoint:

```text
train = 11, 17, 23, 29, 31
dev   = 101, 103, 107
test  = 1009, 1013, 1019
```

Both synthetic dataset families (`failure_recovery` and `branching`) are generated for every seed with four traces per seed.

The split specification itself is not enough to freeze the test because changing the dataset generator could change the generated labels/topology while preserving the seed list. Therefore the generated final-test dataset manifest is independently fingerprinted.

Frozen final-test dataset SHA-256:

```text
8e2062bba82a156141b7d1a51ca6f47e995fe9d37b6e19400beaa7ff482d1ead
```

`ctrag.benchmarks.holdout` recomputes this fingerprint from the current generator and rejects the final test if it no longer matches.

## Development execution

Training/development work may use:

```bash
python -m ctrag.benchmarks.holdout run train \
  --output benchmark-results/train

python -m ctrag.benchmarks.holdout run dev \
  --output benchmark-results/dev
```

Each output directory contains the normal benchmark files plus:

```text
holdout-bundle.json
```

The holdout bundle records:

- split name;
- evaluation status;
- holdout protocol version;
- holdout specification SHA-256;
- generated split dataset SHA-256;
- preregistration identity/hash;
- SHA-256 and byte size for every raw result file.

The command also prints a SHA-256 fingerprint for the bundle manifest itself.

## CI policy

Normal CI runs:

1. unit/integration tests;
2. the historical v0.1 regression benchmark;
3. Phase 2 `train`;
4. Phase 2 `dev`;
5. a **fingerprint-only** calculation for `test`;
6. paper/research artifacts from `dev` only.

CI does **not** run retrieval or aggregate metrics on the final test split.

The fingerprint-only operation verifies that the test dataset has not drifted but does not produce retrieval results:

```bash
python -m ctrag.benchmarks.holdout fingerprint test
```

## Final test remains sealed

Running this command without explicit unblinding fails:

```bash
python -m ctrag.benchmarks.holdout run test
```

The one-shot confirmatory run is intentionally explicit:

```bash
python -m ctrag.benchmarks.holdout run test \
  --output benchmark-results/final-test \
  --unblind-final-test
```

This command must be executed only after:

- Phase 2 preregistration is frozen;
- strong baseline/model selection is complete on train/dev;
- scoring/hyperparameter choices are frozen;
- correctness corrections intended for the confirmatory run are complete;
- the final experiment configuration is fingerprinted.

The pristine final run is **not executed as part of issue #18**. Issue #18 establishes and validates the sealing mechanism before strong-baseline tuning begins.

## Unblinding sentinel

A pristine final-test run writes:

```text
benchmark-results/final-test/UNBLINDED.json
```

That generated sentinel contains:

- frozen holdout specification hash;
- frozen test dataset hash;
- final result-bundle hash;
- status `unblinded`.

Immediately after the first final evaluation, that sentinel must be committed as:

```text
research/holdout/UNBLINDED.json
```

Once the committed sentinel exists, a normal test run is rejected. Every subsequent evaluation must explicitly declare itself a replication:

```bash
python -m ctrag.benchmarks.holdout run test \
  --output benchmark-results/test-replication \
  --replication
```

The resulting bundle is labeled:

```text
evaluation_status = replication
```

This prevents a repeatedly inspected final test from continuing to be described as a pristine holdout.

## Synthetic-holdout limitation

This repository is public and the synthetic generator is deterministic. Consequently, the frozen synthetic holdout is a **procedural anti-tuning boundary**, not a cryptographically secret dataset. Anyone could deliberately reconstruct the test seeds/labels from source code.

For that reason:

- the synthetic test cannot by itself establish external validity;
- issue #19/#20 must introduce real or independent public datasets;
- where a future dataset supports confidential/hidden labels, the same interface should separate public features/manifests from protected final labels;
- the final scientific report must distinguish synthetic holdout evidence from genuinely independent evaluation.

## What counts as test exposure

The following count as final-test exposure:

- executing retrieval over the final test split;
- inspecting final-test aggregate metrics;
- using final-test results to choose models, weights, thresholds or prompts.

Computing the frozen dataset SHA-256 alone does not count as result exposure because it yields no retrieval result or performance aggregate.

## Protocol changes

Changing any of the following after freeze requires a documented protocol amendment before final unblinding:

- split seeds;
- dataset families;
- traces per seed;
- generator semantics that alter the frozen test dataset;
- preregistration primary endpoints or confirmatory evaluation rules.

The old hash must remain in history. A new protocol receives a new version and new fingerprints rather than silently replacing v1.
