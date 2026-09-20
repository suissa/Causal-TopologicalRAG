# Related Work: Temporal Memory, Navigation, Causality and Drift

This note defines novelty boundaries for CT-RAG.

## Zep / Graphiti

Rasmussen et al., **Zep: A Temporal Knowledge Graph Architecture for Agent Memory**, arXiv:2501.13956 (2025).

Zep/Graphiti is prior art for temporally-aware agent memory over a knowledge graph and historical relationships. CT-RAG must not claim bitemporal/temporal graph memory as its primary novelty.

CT-RAG's intended distinction is the combination of execution/runtime causal provenance, explicit causal/temporal/behavioral edge semantics, directed causal retrieval, basins/attractors and a mutable non-authoritative terrain.

Reference: https://arxiv.org/abs/2501.13956

## HippoRAG

Jiménez Gutiérrez et al., **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**, NeurIPS 2024, arXiv:2405.14831.

HippoRAG combines LLMs, knowledge graphs and Personalized PageRank. CT-RAG should therefore avoid novelty language such as "retrieval becomes navigation" by itself.

Reference: https://arxiv.org/abs/2405.14831

## Distributed temporal order

Lamport (1978), **Time, Clocks, and the Ordering of Events in a Distributed System**, introduced the happens-before partial order. Vector clocks (Fidge/Mattern) refine causal/potential-causal ordering in distributed executions.

CT-RAG's invariant is compatible with this tradition: temporal/order evidence constrains what could have caused what, but an `E_t` relation does not automatically become an observed `E_c` relation. Runtime `causation_id` or another explicit provenance source supplies the stronger claim.

## Event time, processing time and watermarks

Akidau et al., **The Dataflow Model** (2015), provides established semantics for unbounded out-of-order processing, event time, processing time and watermarks.

CT-RAG should adopt these semantics for ingestion consistency rather than invent a separate watermark model.

Reference: https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/

## Temporal databases

Snodgrass/TSQL2 and later temporal database work distinguish valid time from transaction/system time. CT-RAG's planned bitemporal semantics are an application of established temporal-database ideas, not a novelty claim.

## Interval-valued state

Allen (1983), **Maintaining Knowledge about Temporal Intervals**, provides the classic interval relations (before, meets, overlaps, during, starts, finishes, equals and converses). CT-RAG should reuse this algebra when states have duration.

DOI: 10.1145/182.358434

## Storage strength vs retrieval strength

Bjork & Bjork's New Theory of Disuse separates storage strength from retrieval strength. This is closely analogous to CT-RAG's distinction:

```text
historical evidence persists
!=
current retrieval/navigation influence
```

The analogy motivates terminology but does not imply neurological equivalence.

## Causal hierarchy

Pearl's hierarchy separates association, intervention and counterfactual reasoning. CT-RAG maps its evidence policies to those levels and must not promote observational topology to intervention/counterfactual claims without additional assumptions.

## Causal discovery from time series

Granger-style methods and PCMCI are candidate tools for proposing `INFERRED` edges from temporal series. Their output should preserve method, assumptions, confidence and evidence and should not be stored with the same authority as runtime-declared event causation.

Relevant overview: Runge et al., *Inferring causation from time series in Earth system sciences*, Nature Communications 10, 2553 (2019).

## Concept drift

ADWIN and CUSUM provide established change-detection approaches. CT-RAG can apply them to streams of transition-probability or absorption-distribution statistics to detect basin/terrain change automatically.

## Novelty boundary

The paper should not claim novelty for any item above in isolation. The candidate contribution is their use inside one execution-memory retrieval model with explicit causal provenance, basin/attractor structure and non-authoritative adaptive terrain.