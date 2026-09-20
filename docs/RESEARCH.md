# Causal-Topological RAG — Research Specification

## 1. Problem statement

Conventional Retrieval-Augmented Generation retrieves external information primarily by lexical or semantic relevance. CT-RAG studies a different problem for stateful/event-driven systems:

> Given a current state or event, retrieve not only memories that resemble the query, but the historical evidence that is structurally, causally and behaviorally relevant to how the system reached or left that state.

The motivating distinction is:

```text
semantic retrieval:  what resembles this?
causal retrieval:     what led to this?
topological retrieval: where is this in the execution terrain?
behavioral retrieval: which trajectories/basins contain comparable experience?
```

CT-RAG does not claim that semantic similarity, temporal order or graph adjacency is itself causality.

CT-RAG is therefore positioned as a **Structured Experiential Memory** layer for long-lived agents rather than as a replacement for vector retrieval or other RAG families. Vector/lexical retrieval can remain the anchor mechanism; CT-RAG adds execution-aware navigation over causal, temporal and behavioral experience.

> **Retrieval becomes navigation; learning becomes terrain modification.**

Here, "learning" means updating non-authoritative navigational influence over preserved historical evidence, not rewriting the event history or claiming that graph navigation itself is novel.

## 2. Memory topology

Let the memory topology be a heterogeneous directed graph

\[
G = (V, E_s, E_c, E_t, E_b)
\]

where:

- \(V\): memories, execution events or state observations;
- \(E_s\): semantic relations;
- \(E_c\): explicit causal relations;
- \(E_t\): temporal-order relations;
- \(E_b\): behavioral/execution-flow relations.

The edge sets are intentionally distinct. In particular:

\[
E_t \not\Rightarrow E_c
\]

A temporal predecessor therefore remains temporal unless independent evidence establishes a causal relation.

## 3. Causal edge

A causal edge is represented as

\[
e_c = (u, v, p, q, \mathcal{E})
\]

where:

- \(u\) is the source;
- \(v\) is the consequence/target;
- \(p\) is provenance class;
- \(q \in [0,1]\) is confidence;
- \(\mathcal{E}\) is the evidence set.

Current provenance classes are:

```text
execution
workflow
dependency
event
inferred
hypothesized
```

The implementation assigns provenance factors for retrieval. These factors are model parameters, not universal causal probabilities.

Observed execution/event/workflow evidence and inferred/hypothesized relationships therefore remain distinguishable throughout storage, traversal and ranking.

## 4. Causal paths

For an anchor \(a\) and candidate \(v\), a causal path is

\[
P(a,v) = (a, x_1, ..., v)
\]

using only edges in \(E_c\).

The current implementation computes path confidence from edge confidence, edge weight and provenance factor. If multiple paths connect the same pair, CT-RAG retains a best explainable path and an aggregate confidence across alternatives.

Temporal and behavioral edges can influence topological retrieval but never contribute to causal-path confidence.

## 5. Navigation signals are not a single metric space

CT-RAG does **not** claim that causal, temporal-hop or behavioral relations form metrics in the mathematical sense. Causal reachability is directed and asymmetric; temporal sequence is directed; behavioral affinity need not satisfy symmetry or the triangle inequality.

We therefore distinguish:

- semantic distance/similarity, when the embedding space supports it;
- directional causal traversal cost, e.g. `h_c^past(a,v)` and `h_c^future(a,v)`;
- temporal clock separation `Δt(a,v) = |t_a - t_v|`;
- temporal-hop count as a directional sequence cost, not a metric;
- behavioral/topological components as navigation priors or affinities.

The reference `EventProjector` fixes the scope of `TEMPORAL` edges to **consecutive events within the same `execution_id`**. It never connects arbitrary adjacent events in the global stream. Temporal-hop counts are therefore execution-relative and do not change merely because unrelated system traffic increases.

Two events can be semantically distant while causally adjacent, or semantically nearly identical while belonging to unrelated executions. CT-RAG keeps those signals heterogeneous rather than pretending they are coordinates in one common metric space.

## 6. Retrieval architecture and baseline score

The preferred CT-RAG formulation is **staged retrieval**, not an assertion that heterogeneous raw distances can be added directly:

```text
query
  -> semantic/lexical anchor generation
  -> topology/basin expansion
  -> directed causal traversal
  -> mode constraints
  -> calibrated or learned reranking
```

The repository retains a weighted linear score as a reproducible baseline and ablation interface:

\[
S_{baseline}(v \mid q,a) =
\alpha S_{semantic}
+ \beta S_{lexical}
+ \gamma S_{causal}
+ \delta S_{topological}
+ \epsilon S_{temporal}
+ \zeta S_{behavioral}
\]

The component functions exposed by the implementation are normalized priors/affinities rather than raw hop counts added directly to cosine similarity. Fixed mode weights are experimental presets, **not** a claim of optimal calibration. An arXiv-grade successor should compare the baseline against learning-to-rank or calibrated reranking while preserving the staged candidate-generation/traversal architecture.

The system currently supports:

- `SIMILAR`;
- `WHY`;
- `WHAT_NEXT`;
- `RECOVERY`;
- `COUNTERFACTUAL`.

`WHY` navigates causal ancestors. `WHAT_NEXT` navigates causal descendants. `RECOVERY` prioritizes observed successful recovery trajectories. `COUNTERFACTUAL` searches historical divergence points but is explicitly labeled observational/hypothesis support.

## 7. Staged retrieval

The research-grade staged path is:

```text
query
  -> global semantic/lexical anchor search
  -> basin/topology expansion
  -> directed causal traversal
  -> mode-specific reranking
  -> explainable ranked context
```

Every retrieval hit can expose its selected anchor, component scores, causal hop count and causal-path evidence.

## 8. Attractors and basins

Given a registered/discovered attractor \(A\), its finite-hop basin is the reverse-reachable set under the configured basin edge kinds:

\[
B_h(A) = \{x \in V : d(x,A) \le h\}
\]

The current implementation is a finite graph construction rather than a claim that every registered attractor satisfies the asymptotic definition used in continuous dynamical-systems theory.

Attractors can be:

- manually/domain declared;
- discovered structural sinks;
- discovered recurrent strongly connected components.

Discovered attractors preserve an explicit `origin` so empirical discovery cannot be confused with domain declaration.

## 9. Dynamic terrain

CT-RAG separates historical truth from navigational influence.

For transition `e`, the terrain keeps the observed transition count `f_e` and a separate navigational influence `w_e^nav` outside the authoritative edge object.

The original reproducible baseline supports frequency reinforcement. Frequency-only reinforcement has an acknowledged popularity bias: common paths can become easier to retrieve merely because they are common.

The implementation now also exposes **surprise-weighted reinforcement**:

\[
\Delta w_e =
\eta
\frac{\min(|\delta_e|,c)}{c}
\frac{1}{\sqrt{1+n_e}}
\]

where `δ_e` is an externally supplied prediction-error/surprise signal, `c` bounds the contribution and `n_e` is the prior observation count. This supports TD-error-style or calibrated-residual signals without claiming that CT-RAG itself estimates a TD error.

Rare-but-critical paths can be protected with a minimum retrieval-strength floor using the existing protected-edge mechanism. Protection changes accessibility, not stored history or causal authority.

Erosion still applies exponential decay to navigational influence without deleting events or rewriting causal provenance/confidence:

\[
w_e^{nav}(t + \Delta t)
= \max(w_{min}, w_e^{nav}(t)e^{-\lambda \Delta t})
\]

For a causal path, the current terrain-aware retriever uses the geometric mean of edge influences as a final multiplicative navigation prior.

```text
historical/storage strength
!=
current retrieval/navigation strength
```

This separation is analogous to Bjork & Bjork's storage-strength versus retrieval-strength distinction. The analogy motivates terminology; CT-RAG does not claim to be a cognitive model.

## 10. Basin drift: structural baseline and distributional target

The implementation currently measures **structural membership drift** with Jaccard distance:

\[
D_J(A) = 1 - \frac{|B_{before}(A) \cap B_{after}(A)|}{|B_{before}(A) \cup B_{after}(A)|}
\]

This answers whether basin membership changed. It does **not** estimate how transition probabilities or absorption probabilities changed, so the paper must not present Jaccard membership drift as a complete dynamical estimator.

The research target is a windowed transition model `P_t`. For attractors `A` in a set `𝒜`, define an empirical absorption distribution `π_t(𝒜|x)` from transitions observed in window `W_t`.

Distributional basin drift can then use total variation or Jensen-Shannon divergence:

\[
D_{TV}(t_i,t_j \mid x) = \frac{1}{2}\sum_{a\in\mathcal{A}} |\pi_{t_i}(a\mid x)-\pi_{t_j}(a\mid x)|
\]

with `D_JS(π_ti || π_tj)` as an alternative. Change-point detectors such as ADWIN or CUSUM should operate on the resulting drift-statistic stream instead of relying on visual inspection.

This distributional estimator is planned work; the current code implements only the Jaccard structural baseline.

## 11. Causality levels

CT-RAG deliberately separates three epistemic levels.

### 11.1 Observed structural causality

A causal relationship recorded explicitly by an execution/workflow/event mechanism, e.g. an event carrying a `causation_id`.

This is evidence about the software execution relation represented by the system. It is not automatically a philosophical or interventionist proof of causation in the external world.

### 11.2 Inferred causal relation

A relation proposed by an inference procedure and stored with `provenance=inferred` plus explicit confidence/evidence.

It is weaker than otherwise equivalent observed execution evidence in the current ranking model.

### 11.3 Counterfactual retrieval and counterfactual hypothesis

Historical branching can support questions such as “where did successful and failed trajectories diverge?” CT-RAG treats this first as a **retrieval problem**.

For a current degraded trajectory such as:

```text
PaymentFailure -> Retry -> Timeout -> HumanIntervention
```

the terrain may preserve an alternative historical branch:

```text
PaymentFailure -> Retry -> ProviderFallback -> Recovered
```

A `COUNTERFACTUAL` query may retrieve that alternative when the historical conditions are semantically/structurally comparable, even if the path currently has low terrain influence because it has eroded through disuse.

The retrieval objective is therefore to identify:

- comparable historical anchor states;
- divergence points;
- alternative observed branches;
- paths reaching a desired attractor such as `Recovered`;
- provenance and conditions attached to those paths.

Dynamic terrain affects retrieval strength, not historical existence. An eroded path can remain queryable; rare critical paths may additionally be protected by a retrieval-strength floor.

This supports hypothesis generation and recovery planning. It does **not** establish the stronger claim that taking the alternative action now would cause recovery. That requires an interventional or otherwise identified causal model.

CT-RAG therefore never promotes a retrieved divergence point or historical alternative into a proven counterfactual effect.

See [`PAPER_POSITIONING.md`](PAPER_POSITIONING.md) for the paper framing and proposed counterfactual-retrieval hypothesis.

## 12. Experimental hypothesis

The primary hypothesis is:

> For diagnostic and recovery queries over stateful execution histories, causal-topological retrieval recovers more causally relevant context with fewer irrelevant memories than semantic nearest-neighbor retrieval alone.

The controlled synthetic benchmark evaluates seven retrieval arms:

```text
lexical only
dense only
dense + lexical
graph/topology only
dense + causal
dense + topological
full CT-RAG
```

with standard retrieval metrics plus causal/trajectory metrics. See [`REPORT.md`](../REPORT.md) for the current controlled proof-of-concept results and its explicit external-validity limitations.

The report is **not** a claim that CT-RAG has already outperformed strong learned dense models, GraphRAG or BasinRAG on shared external datasets.

## 13. Reproducible artifacts

Run:

```bash
python -m ctrag.benchmarks
python -m ctrag.research_artifacts --benchmark-dir benchmark-results --out research-artifacts --k 3
```

The first command produces machine-readable benchmark data. The second derives:

- `topology.dot`;
- `causal-path.dot`;
- `benchmark-k3.md`;
- `manifest.json` with SHA-256 input/output fingerprints.

No benchmark value in those generated files needs to be copied manually.

## 14. Exogenous interventions and quasi-experimental identification

CT-RAG distinguishes ordinary observations from exogenous system changes such as deployments, configuration changes, feature-flag changes and rollbacks.

These events may define intervention boundaries, but the intervention marker itself is not proof of effect. A causal estimate requires an explicit estimand and identification assumptions, for example a justified comparison/control series and parallel-trends diagnostics for difference-in-differences.

A future intervention record should preserve intervention identity/kind, target scope, effective time, pre/post windows, comparison definition, assumptions, diagnostics, effect estimate, uncertainty and provenance back to raw evidence.

This provides a path from descriptive basin drift to quasi-experimental analysis when real exogenous cuts exist, while preserving the distinction between association, intervention and counterfactual claims.

See [`INTERVENTIONS.md`](INTERVENTIONS.md).

## 15. Related work

### Retrieval-Augmented Generation

Lewis et al. introduced RAG as generation combining parametric model memory with explicit non-parametric retrieval. CT-RAG retains external retrieval but focuses on stateful execution memory and heterogeneous structural signals.

Reference: Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020: <https://arxiv.org/abs/2005.11401>

### Zep / Graphiti

Rasmussen et al. introduced Zep/Graphiti as a temporally-aware knowledge-graph architecture for agent memory. CT-RAG therefore does not claim temporal/bitemporal graph memory itself as novel. Its intended distinction is execution-provenance causality, explicit causal/temporal/behavioral edge semantics, directional causal retrieval, basins/attractors and dynamic terrain.

Reference: Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*, arXiv:2501.13956 (2025): <https://arxiv.org/abs/2501.13956>

### HippoRAG

HippoRAG already demonstrates graph-based memory retrieval using knowledge graphs and Personalized PageRank. CT-RAG therefore avoids treating the generic statement "retrieval becomes navigation" as a novelty claim.

Reference: Jiménez Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*, NeurIPS 2024 / arXiv:2405.14831: <https://arxiv.org/abs/2405.14831>
### GraphRAG

Edge et al. proposed graph-based indexing and community summarization to answer global questions over large text corpora. CT-RAG differs in its primary target: execution/event memory in which some graph edges can come from recorded causal structure rather than entity extraction alone.

Reference: Darren Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024: <https://arxiv.org/abs/2404.16130>

### BasinRAG

BasinRAG explicitly frames retrieval through topological structure and dynamical basins. Its official citation identifies the work as *BasinRAG: High-Performance Topological Retrieval-Augmented Generation via Dynamical Basins*, version 1.0.3, released 2026-09-08, DOI `10.5281/zenodo.22664948`.

References:

- code/citation: <https://github.com/Basinfy/BasinRAG>
- DOI: <https://doi.org/10.5281/zenodo.22664948>

CT-RAG takes the basin/topological intuition in a different direction: a stateful/event-driven system can possess explicit execution provenance and causal identifiers that do not need to be reconstructed solely from document adjacency or semantic similarity.

### Temporal-order and streaming foundations

Lamport's happens-before relation and later vector-clock work ground partial ordering in distributed systems. CT-RAG treats temporal order as necessary context but not sufficient evidence for an observed causal edge.

Akidau et al.'s Dataflow Model provides established semantics for event time, processing time, watermarks and out-of-order streams. CT-RAG's ingestion model should reuse these semantics rather than invent an incompatible watermark model.

Allen's interval algebra is the reference model for interval-valued state. Snodgrass/TSQL2 and temporal-database work ground valid-time versus transaction/system-time semantics. These are prior art, not CT-RAG novelty claims.

Bjork & Bjork's storage-strength versus retrieval-strength distinction is a useful analogy for immutable historical evidence versus mutable navigational influence.

Pearl's causal hierarchy grounds CT-RAG's separation between association, intervention and counterfactual claims. PCMCI/Granger-family methods are candidates for `provenance=inferred`, never automatic substitutes for runtime-declared causation.

ADWIN/CUSUM and concept-drift literature ground automatic detection over transition/absorption drift statistics.

Detailed novelty boundaries: [`RELATED_WORK_TEMPORAL_CAUSAL.md`](RELATED_WORK_TEMPORAL_CAUSAL.md).
### Event Sourcing

Event Sourcing records application state changes as a sequence of events and allows state to be rebuilt from the event log. CT-RAG uses that event history as an authoritative source from which retrieval projections can be constructed; it does not make the retrieval graph the source of truth.

Reference: Martin Fowler, *Event Sourcing*, 2005: <https://martinfowler.com/eaaDev/EventSourcing.html>

## 16. Current research limits

The current evidence remains limited by:

- synthetic trace templates;
- deterministic hashing embeddings in the published baseline report;
- a small graph regime;
- known anchors in the controlled ablation;
- absence of a shared real-world benchmark against production learned retrievers;
- absence of interventional causal ground truth.

The next external-validity experiments should add learned dense retrieval, production BM25, real event-sourced traces, unknown-anchor evaluation and shared comparisons where the competing methods can be run under equivalent retrieval budgets.
