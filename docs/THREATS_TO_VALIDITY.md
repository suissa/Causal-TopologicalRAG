# Threats to Validity

This document records known limitations of CT-RAG's current evidence and evaluation design. It is intended to constrain claims, not to weaken reproducibility.

## Construct validity

### Basin discretization

Current experiments frequently map operational outcomes into a finite set of attractors such as `Recovered` and `HumanIntervention`. Real systems may exhibit continuous, overlapping or hierarchical operational states. A discrete basin representation can hide intermediate degradation regimes.

Mitigation:
- report the basin construction rule;
- preserve raw trajectories and transition counts;
- compare structural membership drift with distributional drift;
- evaluate alternative basin definitions where data permits.

### Total Variation as a drift statistic

Total Variation is interpretable and bounded, but it is only one divergence measure. A positive result that disappears under Jensen-Shannon or other reasonable alternatives would indicate estimator sensitivity.

Mitigation:
- sensitivity analysis across detector thresholds;
- planned TV versus JS comparison;
- planned ADWIN/CUSUM change detection.

### Retrieval metrics versus downstream utility

Recall, nDCG, causal-path recall and basin purity measure evidence retrieval quality. They do not establish that a downstream agent generated a correct diagnosis, recovery action or causal conclusion.

Mitigation:
- keep retrieval claims separate from agent-performance claims;
- add end-to-end task evaluation separately.

## Internal validity

### Synthetic event generation

Controlled synthetic traces deliberately encode known structures. Generator assumptions may accidentally make the target topology easier to recover.

Mitigation:
- deterministic fixtures are used only for mechanism/ablation claims;
- gold labels are authored separately from retrieval output;
- future external datasets are evaluated separately rather than pooled with synthetic results.

### Leakage and chronology

A detector evaluated using future windows can manufacture early-warning performance.

Mitigation:
- early-degradation experiments evaluate each window chronologically;
- the healthy baseline is fixed before degradation;
- future observations do not update earlier scores;
- sealed holdout fingerprints are preserved in the science gate.

### Causal-label contamination

Temporal adjacency, span hierarchy or semantic proximity can be mistaken for causality.

Mitigation:
- `TEMPORAL != CAUSAL` is an invariant;
- public workflow step order produces TEMPORAL edges only;
- causal edges require explicit provenance;
- inferred edges remain distinguishable from runtime-declared causation.

### Hyperparameter tuning

A hand-selected TV threshold or sustained-window count may overfit one synthetic scenario.

Mitigation:
- run the full sensitivity grid over thresholds 0.05-0.20 and 1-4 sustained windows;
- report the whole robustness surface, not only the default point;
- future production tuning must use train/dev incidents, never the final test corpus.

## External validity

### Event-driven bias

CT-RAG is designed around stateful/event-driven systems where events, executions, traces or comparable state transitions are observable. Purely synchronous systems with no durable execution identity may require additional instrumentation or a different projection strategy.

### Current external corpus

The current Gate C corpus includes real public GitHub Actions executions. Those traces test transfer to procedural execution histories, but they are not representative of microservice production telemetry.

Mitigation:
- add public microservice/telemetry adapters;
- keep results source-specific;
- never pool GitHub Actions, synthetic incidents and microservice telemetry into a single headline metric.

### Public AIOps datasets

AIOps Challenge 2020 publishes failure records, business metrics, infrastructure metrics and call-chain traces, while SMD from the OmniAnomaly release contains multivariate server-machine telemetry. These can improve realism, but they answer different questions: AIOps2020 can support execution/trace experiments; SMD is more suitable for anomaly/drift baselines than causal-path evaluation.

Mitigation:
- preserve dataset-native labels;
- state which CT-RAG claims each dataset can and cannot test;
- do not manufacture causal edges where source provenance is absent.

## Statistical conclusion validity

### Small sample sizes

Point estimates from a small number of incidents can have high variance.

Mitigation:
- report sample count;
- bootstrap incident-level lead-time statistics once multiple incidents are available;
- report median, quartiles and confidence intervals rather than a single mean.

### Bootstrap interpretation

Bootstrap intervals are only as representative as the sampled incidents. Resampling synthetic variants does not create real-world external validity.

Mitigation:
- the current controlled-cohort bootstrap is labeled synthetic;
- production confidence intervals must use held-out real incidents.

### False-positive cost

An early detector can appear useful by firing constantly.

Mitigation:
- run 30-day stationary healthy null scenarios;
- report scenario-level false-positive rate;
- report mean time between false alarms or, when no false alarm occurs, a lower bound based on observation time.

## Conclusion validity and causal scope

Observed basin drift is descriptive. A deploy/config/feature-flag boundary can support quasi-experimental analysis, but does not by itself identify a causal effect.

Mitigation:
- maintain separate observational, inferred, interventional and counterfactual evidence levels;
- require explicit estimands and assumptions for causal-effect claims;
- preserve intervention provenance.

## Reproducibility threats

External datasets can change, disappear or have ambiguous preprocessing.

Mitigation:
- version adapters and transformation policies;
- fingerprint local inputs;
- store source URLs, dataset/version identifiers and transformation metadata;
- avoid silently downloading mutable data during benchmark runs.

## Claim discipline

The current repository supports:
- controlled mechanism evidence;
- reproducible synthetic ablations;
- limited external procedural-trace validation.

It does not yet support:
- universal superiority over other RAG architectures;
- production forecasting guarantees;
- general causal discovery from telemetry;
- a production-grade counterfactual causal model.