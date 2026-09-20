# Formalization Review Hardening

This note records corrections adopted after adversarial review of the CT-RAG formalization.

## Corrections adopted

1. **No metric claim for directed causal/temporal/behavioral signals.** Causal hops, temporal hops and behavioral affinity are directional traversal costs or navigation priors, not a common metric space.
2. **Temporal-hop scope is fixed.** The reference `EventProjector` creates `TEMPORAL` edges only between consecutive events sharing one `execution_id`. It does not connect global-stream neighbors.
3. **Linear fusion is a baseline, not the research claim.** The preferred architecture is staged candidate generation, topology expansion, directed causal traversal and calibrated/learned reranking. Fixed weighted fusion remains for reproducibility and ablations.
4. **Basin drift has two levels.** Jaccard membership drift is the implemented structural baseline. Distributional drift over windowed transition/absorption distributions (TV/JS plus ADWIN/CUSUM) is the planned estimator.
5. **Frequency reinforcement is not the recommended adaptive policy.** The historical baseline remains reproducible, but `reinforce_by_surprise()` supports bounded surprise/prediction-error reinforcement with diminishing gain. Rare critical edges already support a protected retrieval-strength floor.
6. **Interventions are first-class research objects, not automatic causal proof.** Deploy/config/flag/rollback boundaries can support quasi-experimental designs, but a causal effect still requires an estimand, control/comparison definition and assumptions.

## Claims explicitly not made

- bitemporal memory is not novel to CT-RAG;
- graph navigation for RAG is not novel to CT-RAG;
- temporal precedence is not causal proof;
- structural basin drift is not itself a causal estimate;
- an intervention marker alone does not identify an effect;
- the fixed weighted score is not claimed optimal;
- the current synthetic benchmark does not establish external superiority.

## Proposed novelty target

The narrow contribution CT-RAG should defend is the combination of:

- event/runtime-declared causal provenance;
- explicit separation of causal, temporal and behavioral topology;
- directional retrieval modes over execution memory;
- explainable causal paths;
- basin/attractor organization of execution trajectories;
- non-authoritative dynamic terrain with separate retrieval strength;
- structure-preserving evidence adapters;
- intervention-aware longitudinal analysis of basin/terrain change.

That claim is narrower and more defensible than generic "temporal graph memory" or "retrieval as navigation".