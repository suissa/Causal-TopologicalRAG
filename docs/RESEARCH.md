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

## 5. Distances

CT-RAG can reason over multiple non-equivalent distances:

\[
d_s(i,j) \quad \text{semantic distance}
\]

\[
d_c(i,j) \quad \text{causal hop distance}
\]

\[
d_t(i,j) \quad \text{temporal distance}
\]

\[
d_b(i,j) \quad \text{behavioral/topological distance}
\]

Two events can be semantically distant while causally adjacent, or semantically nearly identical while belonging to unrelated executions.

This is the core reason CT-RAG does not reduce the memory terrain to one embedding metric.

## 6. Retrieval score

The baseline CT-RAG score is a weighted combination:

\[
S(v \mid q,a) =
\alpha S_{semantic}
+ \beta S_{lexical}
+ \gamma S_{causal}
+ \delta S_{topological}
+ \epsilon S_{temporal}
+ \zeta S_{behavioral}
\]

with mode-dependent weights.

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

For transition \(e\), the empirical terrain maintains:

\[
f_e = \text{observed transition count}
\]

and a navigational influence

\[
w_e^{nav}
\]

outside the authoritative edge object.

Repeated trajectories can reinforce \(w_e^{nav}\). Erosion applies exponential decay:

\[
w_e^{nav}(t + \Delta t)
= \max(w_{min}, w_e^{nav}(t)e^{-\lambda \Delta t})
\]

without deleting the event or rewriting causal provenance/confidence.

For a causal path, current terrain-aware retrieval uses the geometric mean of edge influences as a final multiplicative navigation prior.

This implements the conceptual distinction:

```text
historical preservation != current navigational influence
```

## 10. Basin drift

For a basin before and after an update, the current drift metric is Jaccard distance:

\[
D_B(A) = 1 - \frac{|B_{before}(A) \cap B_{after}(A)|}
{|B_{before}(A) \cup B_{after}(A)|}
\]

This makes changes in the empirical terrain measurable between snapshots.

## 11. Causality levels

CT-RAG deliberately separates three epistemic levels.

### 11.1 Observed structural causality

A causal relationship recorded explicitly by an execution/workflow/event mechanism, e.g. an event carrying a `causation_id`.

This is evidence about the software execution relation represented by the system. It is not automatically a philosophical or interventionist proof of causation in the external world.

### 11.2 Inferred causal relation

A relation proposed by an inference procedure and stored with `provenance=inferred` plus explicit confidence/evidence.

It is weaker than otherwise equivalent observed execution evidence in the current ranking model.

### 11.3 Counterfactual hypothesis

Historical branching can support questions such as “where did successful and failed trajectories diverge?” It does **not** establish what would have happened under an intervention.

CT-RAG therefore never promotes a retrieved divergence point to a proven counterfactual effect.

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

## 14. Related work

### Retrieval-Augmented Generation

Lewis et al. introduced RAG as generation combining parametric model memory with explicit non-parametric retrieval. CT-RAG retains external retrieval but focuses on stateful execution memory and heterogeneous structural signals.

Reference: Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020: <https://arxiv.org/abs/2005.11401>

### GraphRAG

Edge et al. proposed graph-based indexing and community summarization to answer global questions over large text corpora. CT-RAG differs in its primary target: execution/event memory in which some graph edges can come from recorded causal structure rather than entity extraction alone.

Reference: Darren Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024: <https://arxiv.org/abs/2404.16130>

### BasinRAG

BasinRAG explicitly frames retrieval through topological structure and dynamical basins. Its official citation identifies the work as *BasinRAG: High-Performance Topological Retrieval-Augmented Generation via Dynamical Basins*, version 1.0.3, released 2026-09-08, DOI `10.5281/zenodo.22664948`.

References:

- code/citation: <https://github.com/Basinfy/BasinRAG>
- DOI: <https://doi.org/10.5281/zenodo.22664948>

CT-RAG takes the basin/topological intuition in a different direction: a stateful/event-driven system can possess explicit execution provenance and causal identifiers that do not need to be reconstructed solely from document adjacency or semantic similarity.

### Event Sourcing

Event Sourcing records application state changes as a sequence of events and allows state to be rebuilt from the event log. CT-RAG uses that event history as an authoritative source from which retrieval projections can be constructed; it does not make the retrieval graph the source of truth.

Reference: Martin Fowler, *Event Sourcing*, 2005: <https://martinfowler.com/eaaDev/EventSourcing.html>

## 15. Current research limits

The current evidence remains limited by:

- synthetic trace templates;
- deterministic hashing embeddings in the published baseline report;
- a small graph regime;
- known anchors in the controlled ablation;
- absence of a shared real-world benchmark against production learned retrievers;
- absence of interventional causal ground truth.

The next external-validity experiments should add learned dense retrieval, production BM25, real event-sourced traces, unknown-anchor evaluation and shared comparisons where the competing methods can be run under equivalent retrieval budgets.
