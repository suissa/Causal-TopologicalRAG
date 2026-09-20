# Early Behavioral Degradation Experiment

## Research question

> Can basin drift quantify production-performance degradation before traditional infrastructure alerts fire?

The current experiment is a deterministic mechanism test. It is designed to establish a reproducible measurement protocol, not to claim general production forecasting performance.

## Operational hypothesis

A critical operation has two terminal attractors:

```text
Retry -> Recovered
Retry -> Timeout -> HumanIntervention
```

During healthy operation, most trajectories converge to `Recovered`. Degradation begins when the absorption distribution shifts toward `HumanIntervention`, even while a conventional infrastructure metric remains below its alert threshold.

## Estimator

For each chronological daily window, estimate the attractor absorption distribution:

[
\pi_t =
[P_t(Recovered), P_t(HumanIntervention)]
]

A healthy baseline distribution `pi_0` is estimated from the first five windows.

Behavioral drift is measured with Total Variation:

[
D_{TV}(\pi_t,\pi_0)
= \frac{1}{2}\sum_a |\pi_t(a)-\pi_0(a)|
]

The baseline detector requires two consecutive post-baseline windows with TV drift >= 0.10.

The traditional detector is independent: it fires when an infrastructure metric crosses 0.85.

## Controlled scenario

The fixture starts at:

```text
Recovered          90%
HumanIntervention  10%
```

and gradually shifts toward:

```text
Recovered          50%
HumanIntervention  50%
```

while the infrastructure metric rises more slowly.

In the deterministic fixture:

```text
behavioral alert      day 8
infrastructure alert  day 10
lead time             2 days
```

This positive lead time is a property of the controlled fixture only.

## Why this is a CT-RAG experiment

The signal is not "CPU is high" or "latency crossed a limit."

It is:

> the probability mass of observed execution trajectories is moving from one attractor basin toward another.

That makes the monitored object a behavioral topology rather than a single infrastructure series.

## Reproducibility

Run:

```bash
python -m ctrag.benchmarks.early_behavioral_degradation \
  --output benchmark-results/early-behavioral-degradation
```

Artifacts:

```text
early-behavioral-degradation.json
early-behavioral-degradation.csv
README.md
manifest.json
```

The manifest fingerprints both source and results.

## Current limitations

- synthetic deterministic trajectories;
- two terminal attractors only;
- fixed healthy baseline;
- daily non-overlapping windows;
- fixed TV threshold;
- two-consecutive-window detector rather than ADWIN/CUSUM;
- no uncertainty interval yet;
- no infrastructure noise model;
- no real incident data;
- no claim that the detector is causal or predictive outside this fixture.

## Next validation stage

The next experiment should use real or replayed production traces and compare:

1. time-to-detection from basin/absorption drift;
2. time-to-detection from latency/error-rate/resource alerts;
3. false-positive rate;
4. false-negative rate;
5. detection lead-time distribution;
6. robustness across window sizes and drift thresholds;
7. ADWIN/CUSUM versus the fixed sustained-threshold baseline.

A publishable result requires positive lead time across held-out incidents without unacceptable false-alert cost.
