---
title: "CT-RAG: Structured Experiential Memory via Causal-Topological Retrieval for Long-Lived Agents"
author:
  - "Jean Carlo Nascimento"
affiliation: "AllasCode Institute"
date: "September 2026"
bibliography: ctrag-references.bib
link-citations: true
---

# Candidate titles

1. **CT-RAG: Structured Experiential Memory via Causal-Topological Retrieval for Long-Lived Agents** *(selected)*
2. **Beyond Similarity and Recency: Runtime Causal Provenance for Long-Lived Agent Memory**
3. **Retrieval Becomes Navigation: Causal-Topological Memory and Dynamic Terrain for Agents**

# Abstract

Long-lived software agents accumulate execution histories that are difficult to use as memory. Conventional retrieval-augmented generation (RAG) primarily ranks items by semantic similarity, while temporal memory systems emphasize when facts were valid or observed. Neither geometry is sufficient when the relevant evidence is defined by runtime execution structure: an event can be causally authoritative despite being semantically dissimilar, temporally distant, or received out of order. We introduce **CT-RAG**, a retrieval architecture for **structured experiential memory** that represents execution evidence using typed causal, temporal, and behavioral relations, separates authoritative causal provenance from inferred association, and maintains a dynamic, non-authoritative **terrain** whose navigational influence can be reinforced or eroded without rewriting immutable historical evidence. Retrieval is therefore treated as navigation over preserved experience, while learning changes navigation rather than truth.

We evaluate CT-RAG using controlled adversarial fixtures designed to separate causal structure from simpler ranking signals. In an out-of-order causation experiment, a child event is ingested before its parent, with a 336-hour event-time gap and 25 semantically adversarial distractors. Semantic-only and recency-only retrieval rank the true parent 26th of 26 candidates, while CT-RAG `WHY` ranks it first using explicit runtime `causation_id` provenance. In an eroded-path recovery experiment, a raw-success baseline prefers an unreachable `GlobalFix` path with 9/10 historical success, whereas CT-RAG excludes it because it is not causally reachable from the current state and ranks the best reachable recovery first. Conservative small-sample support is represented by the 95% Wilson lower bound rather than an ad hoc count bonus. CT-RAG achieves Precision@1 = 1.0 in this controlled recovery fixture, while terrain-only, recency-only, and raw-success baselines each obtain 0.0; false-rescue increases from 0.0 at k=1 to 0.75 at k=4, exposing rather than hiding degradation as more alternatives are returned. A secondary same-logic degradation study under the current zero-FPR null yields a median +1 day relative lead across a 28-cell parameter surface, with the positive-lead cells forming one contiguous region and the +1-day lead invariant to four phase offsets. These experiments are mechanism-level synthetic evidence, not production generalization. The results support a narrower claim: **runtime causal reachability contains retrieval information that semantic similarity, recency, and global success statistics do not encode**.

# 1. Introduction

Retrieval-augmented generation augments a parametric language model with information retrieved from an external memory [@lewis2020rag]. Most deployed and research RAG systems begin from a geometric assumption: if two items are close in an embedding space, one is likely to be useful for answering a query. Long-term agent-memory systems add persistence, reflection, hierarchical storage, temporal knowledge graphs, or graph navigation [@park2023generative; @packer2023memgpt; @rasmussen2025zep; @gutierrez2024hipporag]. This progression is important, but agents operating inside event-driven systems face an additional problem. Their relevant memory is often not merely *about the same thing* or *nearby in time*. It is the execution that caused the current event, the previously successful route reachable from the current state, the basin toward which repeated behavior is drifting, or the historical branch whose current navigational influence has eroded without invalidating the evidence that it once occurred.

Distributed systems make the distinction concrete. Event arrival order is not event time, event time is not causal order, and semantic similarity is not causal authority. Lamport's happens-before relation established the importance of partial order over physical-clock proximity in distributed computation [@lamport1978time]. Stream-processing systems similarly distinguish event time from processing time because out-of-order arrival is intrinsic rather than exceptional [@akidau2015dataflow]. Temporal databases distinguish valid time from transaction time for related reasons [@jensen2018temporal]. CT-RAG adopts the same discipline for agent memory: **different relations carry different semantics and must not silently collapse into one another**.

This paper is organized around two distinct functions of experiential topology. **RQ1 treats topology as a relation of authority:** when several memories are semantically or temporally plausible, which prior event is entitled to explain the current event because the runtime explicitly identifies it as causal provenance? **RQ2 treats topology as a relation of feasibility:** among historically successful recovery routes, which alternatives are actually reachable from the current state? Similarity, recency, and global success statistics can rank observations, but they do not encode either authority or state-conditioned feasibility. The two primary experiments are designed to isolate these two roles separately.

We use the term **structured experiential memory** to denote memory organized around observed execution experience rather than only facts or text fragments. This term is intentionally descriptive rather than a claim that graph-structured experience is unique to CT-RAG. Recent work explicitly studies graph-based agent memory and experiential memory [@yang2026graphmemory; @hu2025memorysurvey; @dai2026gsem], while other 2026 systems explore retrieval-driven reconsolidation and selective forgetting [@song2026realm; @rusu2026selective]. Our claim is narrower: CT-RAG combines runtime-owned causal provenance, typed separation of experiential relations, execution basins/attractors, and a dynamic terrain overlay that changes navigational influence without changing historical truth.

The design follows one central invariant:

> **Observed adjacency may create temporal or behavioral evidence, but only explicit or separately justified provenance may create a causal edge.**

This distinction also separates CT-RAG from causal inference. A runtime field such as `causation_id` is not a statistical estimate that one variable causes another; it is provenance asserted by the executing system. Conversely, observational alternatives retrieved from history do not identify intervention effects in Pearl's sense [@pearl2009causality]. Granger-style predictability and PCMCI-style causal discovery can be useful analytical tools, but they answer different questions and are not substituted for runtime provenance [@granger1969causal; @runge2020pcmci].

This paper makes four contributions.

1. **A typed experiential topology.** We model agent/system experience as nodes connected by causal, temporal, behavioral, and optionally semantic relations, with causal edges carrying explicit provenance and temporal relations carrying scope.
2. **A non-authoritative dynamic terrain.** Repeated trajectories reinforce navigational influence and unused paths erode, while the underlying nodes, edges, provenance, and observed transition history remain intact.
3. **Mode-aware causal-topological retrieval.** `WHY` retrieval can prioritize authoritative causal ancestry even under adversarial semantic and temporal geometry; `RECOVERY` uses causal reachability as an eligibility gate and conservative historical success support to order reachable alternatives without letting low terrain influence act as a veto.
4. **Adversarial controlled evidence.** We construct baselines intended to defeat the method: 25 semantically near and temporally recent distractors in the causal-gap experiment, and a globally higher-success but causally unreachable recovery in the eroded-path experiment. These fixtures isolate information carried by runtime topology rather than merely showing that an implementation satisfies its own specification.

The remainder of the paper describes the model, implementation invariants, experimental protocol, results, and limitations. All numerical results reported here were regenerated by GitHub Actions CI run **#210** from commit **`d2c73de6ae1a691ad7d61c9ece26a4e84c93c44c`** on Python 3.11, 3.12, and 3.13; the tables below use the Python 3.13 uploaded benchmark artifact.

# 2. Related Work

## 2.1 Retrieval-augmented generation and graph retrieval

RAG combines parametric generation with non-parametric retrieval [@lewis2020rag]. Dense retrieval is effective when relevance correlates with semantic similarity, but it does not by itself encode execution causality, temporal validity, or reachability constraints.

HippoRAG combines LLM extraction, a knowledge graph, and Personalized PageRank to support efficient multi-hop retrieval and long-term knowledge integration [@gutierrez2024hipporag]. Its contribution demonstrates that graph structure can improve retrieval beyond flat vector search. CT-RAG does not claim generic graph navigation as novel. The distinction is the graph's semantics: CT-RAG's primary structure is **runtime experience**, and its causal relations are admitted under provenance constraints rather than created because entities co-occur or are semantically associated.

Topo-RAG argues that hybrid enterprise documents should preserve intrinsic data topology rather than linearizing every structure into text; it routes narrative and tabular evidence through structure-appropriate retrieval [@dantart2026toporag]. CT-RAG uses *topology* in a different sense. Topo-RAG preserves the intrinsic spatial/structural topology of an artifact; CT-RAG models the topology of **execution experience** across events and states. An important invariant follows: intrinsic artifact adjacency, temporal adjacency, behavioral continuity, and causal evidence are separate relations.

## 2.2 Long-term agent memory

Generative Agents stores a stream of natural-language experiences, retrieves memories dynamically, and synthesizes reflections that influence future behavior [@park2023generative]. MemGPT manages multiple memory tiers using an operating-system analogy to extend effective context over long interactions [@packer2023memgpt]. Zep/Graphiti uses a temporally aware knowledge graph to dynamically integrate conversational and business data while preserving historical relationships [@rasmussen2025zep]. These systems establish long-term memory as a first-class agent primitive and motivate explicit treatment of memory dynamics.

Recent surveys distinguish factual, experiential, working, structural, and graph-based memory, showing that graph structure is now a broad design family rather than a differentiating claim by itself [@hu2025memorysurvey; @yang2026graphmemory]. GSEM, for example, encodes multi-agent coordination episodes as heterogeneous relational graphs and retrieves structurally similar experience to accelerate adaptation [@dai2026gsem]. REALM reorganizes a heterogeneous memory graph based on retrieval feedback [@song2026realm]. Selective Forgetting reports a useful negative result: a graph-based conversational memory pipeline did not automatically beat a flat vector baseline under matched retrieval budget, while selective pruning could reduce memory with little measured loss [@rusu2026selective]. These findings reinforce our claim discipline: **CT-RAG does not assert that graphs are intrinsically better than vectors.** Its experiments instead ask whether specific runtime relations carry information absent from semantic or temporal baselines.

## 2.3 Time and causality in distributed systems

Lamport formalized happens-before as a partial order independent of physical clock equality [@lamport1978time]. Modern dataflow systems distinguish event time, processing time, and completeness/watermarks to handle unbounded out-of-order streams [@akidau2015dataflow]. Temporal database research similarly distinguishes valid time and transaction time [@jensen2018temporal]. CT-RAG draws an engineering lesson from this literature: memory systems should not treat one scalar timestamp as a universal relation.

CT-RAG further distinguishes **runtime causal provenance** from **causal inference**. Pearl's structural causal models and intervention semantics concern identification of causal effects [@pearl2009causality]. Granger causality concerns predictive temporal precedence [@granger1969causal], while PCMCI/PCMCI+ estimates lagged and contemporaneous causal relations from observational time series under explicit assumptions [@runge2020pcmci]. CT-RAG may store hypotheses from such methods with lower epistemic status, but an event relation asserted by `causation_id` is a different object: it is provenance emitted by the runtime, not an observational causal estimate.

## 2.4 Memory strength as a limited analogy

Bjork and Bjork's New Theory of Disuse distinguishes storage strength from retrieval strength [@bjork1992disuse]. CT-RAG's separation between preserved history and current navigational influence is analogous in a limited engineering sense: a path can become hard to retrieve without being erased. We do **not** claim cognitive or psychological isomorphism. The terrain is an engineered retrieval overlay, not a model of human memory.

# 3. Problem Definition

Let an experiential memory be a typed directed graph

\[
G = (V, E_c, E_t, E_b, E_s),
\]

where \(V\) contains observed events or states; \(E_c\) contains causal relations; \(E_t\) temporal relations; \(E_b\) behavioral/execution-continuity relations; and \(E_s\) optional semantic relations. Semantic and lexical similarity may also be computed as retrieval channels without materializing \(E_s\).

A causal edge is represented as

\[
e_c = (u,v,p,q,\mathcal{E}),
\]

where \(p\) is a provenance class, \(q\in[0,1]\) a confidence, and \(\mathcal{E}\) evidence pointers. The implementation distinguishes provenance such as event, execution, workflow, dependency, inferred, and hypothesized. The epistemic invariant is that an edge in \(E_c\) cannot exist without causal provenance.

A temporal edge is

\[
e_t=(u,v,\sigma),
\]

where \(\sigma\) is a temporal scope. Implemented scopes are `EXECUTION`, `ACTOR_OR_AGGREGATE`, `DEPLOYMENT`, `INCIDENT_WINDOW`, and `GLOBAL_OBSERVED`. The fail-safe default for temporal traversal is execution-local; callers must explicitly opt into broader scope traversal. This prevents accidental chaining of unrelated executions merely because timestamps are adjacent.

For a query \(q\), anchor set \(A\), and mode \(m\), the base retriever combines semantic, lexical, causal, topological, temporal, and behavioral components:

\[
S_{base}(v\mid q,A,m)=
\sum_j w^{(m)}_j s_j(v\mid q,A),
\]

where the weight vector depends on query mode. CT-RAG currently exposes `SIMILAR`, `WHY`, `WHAT_NEXT`, `RECOVERY`, and `COUNTERFACTUAL`. `COUNTERFACTUAL` is observational retrieval of historical divergence and must not be interpreted as an identified intervention effect.

# 4. Architecture and Invariants

## 4.1 Event projection and authoritative causation

The `EventProjector` maps authoritative event records into memory nodes. When event \(v\) declares `causation_id = u` and \(u\) already exists, the projector creates

\[
u \xrightarrow{CAUSAL,\ EVENT} v
\]

with evidence source `event.causation_id`. If \(u\) has not yet arrived, the child is placed in a pending-causation structure keyed by the parent identifier. When \(u\) later arrives, the edge is reconciled deterministically. Arrival order therefore affects when the edge can be materialized, but not its eventual causal authority.

Events sharing an execution identifier may also receive temporal and behavioral edges according to execution sequence. Crucially, sequence alone never creates a causal edge. This prevents the common error of converting "B happened after A" into "A caused B."

## 4.2 Degraded mode without causal provenance

External telemetry often lacks authoritative causal fields. CT-RAG therefore operates in a degraded but explicit mode: trace hierarchy, temporal adjacency, and behavioral continuity may be represented, but disconnected spans or events are not promoted into \(E_c\) merely because no competing explanation exists. This fail-safe behavior reduces recall of causal paths when provenance is missing, but preserves epistemic integrity.

## 4.3 Observability as an evidence contract, not a causal oracle

CT-RAG requires **observable execution context**, but it does not require "perfect telemetry" and it does not treat observability as an oracle that manufactures causality. This distinction matters operationally.

Modern distributed systems already propagate execution context to make traces reconstructable. W3C Trace Context standardizes portable `traceparent`/`tracestate` propagation specifically so requests can be correlated across distributed components [@w3c2021tracecontext]. OpenTelemetry likewise treats traces, metrics, and logs as the basic signals required to understand a running system, and its current Generative AI conventions instrument agent invocations, model calls, and tool executions [@otel2026observability; @otel2026genai]. CT-RAG consumes this class of evidence but assigns different epistemic roles to different fields.

| Evidence available | CT-RAG role | If absent |
|---|---|---|
| execution/trace/request identifiers | behavioral continuity and correlation | lower cross-component recall |
| event time and ingest/observed time | scoped temporal ordering | weaker temporal navigation |
| logs, metrics, traces, tool-call records | typed contextual evidence | reduced diagnostic context |
| explicit `causation_id`, declared workflow dependency, or equivalent runtime provenance | authoritative causal edge | **no causal promotion** |
| full prompts/completions | optional diagnostic evidence | no loss of causal authority by itself |

The final row is important for cost and privacy: a professional observability stack need not record every token or payload. OpenTelemetry itself treats verbose GenAI content as opt-in because it may be large or sensitive. What CT-RAG needs for authoritative causal retrieval is not maximum telemetry volume; it needs **semantically typed provenance and context propagation**.

This also changes how the legacy-system limitation should be stated. Retrofitting a system that emits only unstructured logs can be expensive, and causal coverage will initially be low. That is a real deployment cost. However, missing provenance causes CT-RAG to **abstain from causal authority**, not to infer causality from temporal adjacency. The failure mode is reduced causal recall, not silent conversion of correlation into causation.

## 4.4 Basins and attractors

CT-RAG models repeated execution as movement through topology. A structural sink is a node with no outgoing basin-relevant edge. A recurrent strongly connected component (SCC) may also act as an attractor. Basins are defined as bounded incoming neighborhoods of registered attractors over causal and behavioral relations.

Given two snapshots with basin member sets \(B_a^{(0)}\) and \(B_a^{(1)}\) for attractor \(a\), basin drift is represented by Jaccard distance:

\[
D_B(a)=1-\frac{|B_a^{(0)}\cap B_a^{(1)}|}{|B_a^{(0)}\cup B_a^{(1)}|}.
\]

This is a descriptive topological drift measure, not a causal-effect estimate.

# 5. Dynamic Terrain

The graph records historical structure; the **DynamicTerrain** records current navigational influence. For each edge \(e\), terrain maintains an observation count \(n_e\) and influence \(I_e\). Observing a transition reinforces only the overlay:

\[
I_e \leftarrow \min(I_{max}, I_e + \alpha).
\]

After elapsed time \(\Delta t\), influence decays exponentially:

\[
I_e(t+\Delta t)=\max(I_{min,e}, I_e(t)e^{-\mu\Delta t}).
\]

Protected rare-but-critical edges may use a non-zero floor \(I_{min,e}\). Decay never removes the underlying node, edge, causal evidence, or transition count.

For a causal path \(\pi=(e_1,\ldots,e_k)\), path influence is the geometric mean

\[
I(\pi)=\left(\prod_{i=1}^{k} I_{e_i}\right)^{1/k}.
\]

Outside recovery mode, terrain can act as a multiplicative navigational reranker. In `RECOVERY`, however, low influence is deliberately **not** a veto: an old but historically successful reachable route may need to be surfaced precisely because it has fallen out of current use.

The implementation also supports a transition-surprise signal using a Laplace-smoothed empirical outgoing probability,

\[
\mathrm{surprise}(e)=1-P(e\mid source(e),kind(e)),
\]

but surprise-based reinforcement is not required for the primary results in this paper.

# 6. Recovery Ranking: Reachability and Conservative Historical Support

Recovery retrieval answers a different question from similarity search: *from the current state, which historically observed routes can reach a desired recovered state, and which of those routes have credible evidence of success?*

CT-RAG first requires causal reachability from the current anchor. A successful historical terminal with no causal path from the current state is ineligible, regardless of global success rate. For a reachable terminal, the final branch immediately preceding that terminal provides counts \(s\) successful outcomes among \(n\) observed outcomes.

Rather than rank by the empirical ratio \(\hat p=s/n\) alone, CT-RAG uses the lower endpoint of the 95% Wilson score interval:

\[
LB_W(s,n)=
\frac{\hat p+\frac{z^2}{2n}-z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}
{1+\frac{z^2}{n}},\qquad z=1.96.
\]

This folds small-sample uncertainty into the ranking without inventing an arbitrary support multiplier. For example, \(LB_W(1,1)\approx0.207\), while \(LB_W(8,8)\approx0.676\); both empirical rates are 1.0, but the latter has substantially stronger support.

The terrain-aware recovery reranker retains `terrain_influence` for explanation but adds conservative historical evidence only to causally reachable successful terminals. Conceptually,

\[
S_{REC}(v)=S_{base}(v)+\lambda\,LB_W(s_v,n_v),
\]

subject to

\[
Reachable(A,v)=1.
\]

The current implementation uses \(\lambda=0.35\). This coefficient is an implementation parameter, not claimed as universally optimal.

# 7. Experimental Method

We separate **comparative experiments** from **property/mechanism tests**. Comparative experiments include competing conditions intended to succeed under plausible alternative assumptions. Property tests verify invariants and regression behavior but are not presented as evidence of superiority.

All primary results below are synthetic controlled fixtures. They are designed for causal identification of *mechanism behavior* inside the software artifact, not for estimating production effect sizes.

## 7.1 RQ1: Can runtime causal provenance recover evidence that semantic and recency geometry suppress?

We ingest child event `Action_B_Success` first with explicit

```
causation_id = Action_A_Intent
```

while the parent is absent. We then ingest 25 unrelated events whose type is deliberately adversarial to semantic retrieval: `Action.B.Success.Context`. The parent finally arrives with event time 336 hours earlier than the child.

We compare:

- **Semantic-only:** global dense retrieval over all nodes.
- **Recency-only:** descending event-time ordering.
- **CT-RAG WHY:** mode-aware retrieval anchored at `Action_B_Success`, with causal provenance available from the reconciled `causation_id` edge.

The primary metric is absolute rank of the true parent. The test oracle requires CT-RAG parent rank = 1 and both baselines to place the parent outside the top 10.

## 7.2 RQ2: Does causal reachability add information beyond terrain, recency, and raw historical success?

The current state is `retry`. Reachable historical recovery branches are:

- `ProviderFallback`: 8/10 successful outcomes;
- `ManualPatch`: 1/10;
- `ScriptPatch`: 2/10;
- `CacheReset`: 1/4.

A separate `GlobalFix` branch has 9/10 historical success but originates from a different root and is **not causally reachable from `retry`**.

All historical recovery branches are eroded, while the current terrain is dominated by `Timeout -> HumanIntervention`.

We compare:

- **Terrain-only:** current path influence;
- **Recency-only:** most recent recovered terminal first;
- **Raw-success-global:** global empirical \(P(success)\), ignoring reachability;
- **Structural reachable ceiling:** all recovered terminals causally reachable from `retry`, deliberately treated as an unranked candidate set;
- **CT-RAG RECOVERY:** reachability gate plus Wilson-supported historical success.

The gold top-1 recovery is `ProviderFallback`. We report Precision@1 and false-rescue@k for \(k\in\{1,2,3,4\}\). The broader false-rescue curve matters because a healer consuming multiple suggestions is exposed to harmful alternatives even when top-1 is correct.

## 7.3 RQ3: Does the behavioral signal lead an infrastructure signal under the same detection logic?

This secondary experiment uses daily windows whose terminal absorption distribution shifts gradually from `Recovered` toward `HumanIntervention`. Behavioral drift is total variation from a fixed healthy baseline. To avoid the original unfair comparison against a frozen infrastructure threshold, both behavioral and infrastructure signals use the same sustained-change detection pattern, and the infrastructure threshold is calibrated on stationary null scenarios. In the current generator, however, both achieved FPRs remain exactly zero across the tested grid, so this calibration does not establish a non-zero matched operating point.

The robustness grid contains 28 combinations of behavioral drift threshold and sustained windows. We report the contiguous positive-lead region, lead distribution, FPR gap, and phase-offset sensitivity. The phase analysis shifts window boundaries by fractions 0, 0.25, 0.50, and 0.75. This is an alignment test, not sub-daily evidence.

## 7.4 Mechanism/property validation

Additional CI-gated tests validate that:

- decay changes influence but preserves graph history;
- newly observed recurrent cycles increase SCC internal observations and confidence;
- out-of-order causal reconciliation is deterministic;
- execution-local temporal traversal is the fail-safe default.

These support implementation correctness but are not counted as comparative experimental wins.

# 8. Results

## 8.1 Out-of-order causal provenance

**Table 1. Absolute rank of the true causal parent among 26 total candidates under adversarial semantic and temporal geometry.**

| Method | True parent rank | Top-10 recall | Information used |
|---|---:|---:|---|
| Semantic-only dense retrieval | 26 | 0 | semantic similarity |
| Recency-only | 26 | 0 | event-time recency |
| CT-RAG `WHY` | **1** | **1** | runtime causal provenance + topology |

The reconciled edge carries provenance `EVENT`, evidence source `event.causation_id`, and causal component 0.9 in the retrieved parent hit. The 336-hour temporal gap and 25 semantically close distractors therefore do not prevent the parent from ranking first.

This experiment establishes a specific claim rather than generic retrieval superiority. Semantic similarity and recency fail because the fixture is constructed so that both geometries point toward distractors. CT-RAG succeeds because the runtime supplies a relation those baselines do not represent. The result is therefore evidence that **runtime causal provenance can carry retrieval information orthogonal to semantic and temporal proximity**.

### First-page figure

We propose Figure 1 as a three-panel visual occupying the top half of the first page after the abstract.

- **Panel A — Semantic geometry:** `Action_B_Success` at the center of a dense cluster of 25 `Action.B.Success.Context` distractors; `Action_A_Intent` visually distant. Label: *semantic-only parent rank = 26*.
- **Panel B — Recency geometry:** horizontal event-time axis showing `Action_A_Intent` 336 h to the left, `Action_B_Success`, then the 25 recent distractors. Label: *recency-only parent rank = 26*.
- **Panel C — Runtime provenance:** collapse the same observations into a causal view with a single highlighted `Action_A_Intent --causation_id--> Action_B_Success` edge crossing the temporal gap. Label: *CT-RAG WHY parent rank = 1*.

The caption should state: **Same evidence, three geometries. Similarity and recency concentrate on adversarial distractors; explicit runtime provenance identifies the causal parent.**

## 8.2 Eroded-path rescue

The global raw-success baseline ranks `GlobalFix` first because its empirical success is 0.90, above `ProviderFallback` at 0.80. However, `GlobalFix` is not reachable from `retry`. The structural baseline correctly excludes it but does not order the remaining candidates.

**Table 2. Recovery candidates and conservative historical support.**

| Candidate | Raw success | n | Wilson lower 95% | Reachable from `retry` | CT-RAG rank |
|---|---:|---:|---:|---:|---:|
| ProviderFallback | 0.80 | 10 | **0.490** | Yes | **1** |
| ScriptPatch | 0.20 | 10 | 0.057 | Yes | 2 |
| CacheReset | 0.25 | 4 | 0.046 | Yes | 3 |
| ManualPatch | 0.10 | 10 | 0.018 | Yes | 4 |
| GlobalFix | **0.90** | 10 | **0.596** | **No** | excluded |

The order of `ScriptPatch` and `CacheReset` illustrates why empirical point rate alone is not the recovery score: Wilson support for 2/10 (0.057) exceeds 1/4 (0.046), despite the latter's higher raw point estimate. `ProviderFallback` remains clearly separated: its lower bound is 0.490, a gap of 0.433 from `ScriptPatch` and 0.444 from `CacheReset`.

**Table 3. Top-1 recovery comparison.**

| Method | Output | Precision@1 | Precision@4 |
|---|---|---:|---:|
| Terrain-only | HumanIntervention | 0.0 | — |
| Recency-only | ManualPatch | 0.0 | — |
| Raw success, global | GlobalFix | 0.0 | — |
| Structural reachable ceiling | unranked set of four | n/a | 0.25 |
| CT-RAG `RECOVERY` | **ProviderFallback** | **1.0** | 0.25 |

The top-1 result alone is insufficient for a healer that may consume multiple suggestions. The measured false-rescue curve is therefore reported directly:

\[
FR@1=0.00,\quad FR@2=0.50,\quad FR@3=0.667,\quad FR@4=0.75.
\]

This degradation is not hidden or thresholded away. It shows that the current mechanism is strong at selecting the first candidate in this fixture but does not solve the broader problem of returning a clean multi-candidate set. A production healer should therefore either consume top-1 under stricter confidence policy or add further constraints before acting on lower-ranked suggestions.

The central inference from this experiment is again narrow: **global historical success is insufficient when the highest-success route is not reachable from the current state.** The causal topology supplies an eligibility relation not present in terrain, recency, or success statistics.

### Reachability-gate isolation check

After the numerical evidence freeze used for Tables 1--3, we executed the preregistered mutation that changes only feasibility: an edge `retry -> global_fix` is added while the terminal success statistics remain 9/10 for `GlobalFix` and 8/10 for `ProviderFallback`. Under this mutation, both candidates are causally reachable and `GlobalFix` becomes the top-ranked recovery, with its Wilson lower bound (0.596) exceeding `ProviderFallback` (0.490). The test passed in GitHub Actions CI **#213**, head commit **`9ed7878813b951ca04428e27868d5a4427e8011f`**, on Python 3.11, 3.12, and 3.13.

This isolation check is intentionally reported separately from the CI #210 numerical freeze: it adds no new headline effect size. Its role is falsification-oriented. The observed rank flip supports the interpretation that causal reachability acts as an eligibility gate rather than as an undeclared continuous penalty hidden inside the recovery score.

## 8.3 Secondary result: same-logic early degradation under a zero-FPR null

The original fixed infrastructure threshold produced an apparent +2-day lead at the default behavioral configuration. That comparison was demoted because it compared unlike detector operating rules. The stronger ablation applies the **same sustained-change detector logic** to behavioral and infrastructure signals and calibrates the infrastructure threshold on the same null generator. At \(\theta=0.10\) with two sustained windows, behavioral change is detected on day 8 and infrastructure change on day 9, yielding a **+1-day relative lead**.

The current null generator, however, yields FPR = 0 for both detectors throughout the tested grid. Consequently, the observed FPR gap of 0 is **degenerate**: it does not demonstrate informative matching at a non-zero operating point. The protection of the +1-day result comes from comparing the same detector logic and from its robustness across parameter/phase perturbations, not from evidence that non-zero false-positive rates have been successfully matched.

Across the full 28-cell grid:

- 16 cells have positive lead;
- achieved FPR is 0 for both detectors in all 28 cells, so the FPR gap is 0 but non-informative as an operating-point match;
- the 16 positive cells form one contiguous region in the tested parameter lattice;
- median lead across the 28 cells is +1 day;
- the lead distribution ranges from -1 to +3 days.

The positive region is:

\[
\begin{aligned}
&\theta=0.05, &&s=1..4\\
&\theta=0.075,&&s=1..3\\
&\theta=0.10, &&s=1..3\\
&\theta=0.125,&&s=1..3\\
&\theta\in\{0.15,0.175,0.20\},&&s=1.
\end{aligned}
\]

This shape matters more than the fraction 16/28: positive lead persists under stricter sustained-window requirements only at lower drift thresholds, indicating a modest signal that requires sensitivity rather than a large, obvious separation.

The phase-offset ablation produces paired leads `[1, 1, 1, 1]` for phase fractions 0, 0.25, 0.50, and 0.75. At 0.50 and 0.75, both detectors move one day later, while their difference remains +1. We therefore do **not** claim an absolute detection date such as "day 8" as phase invariant. The stable quantity is the relative lead under this synthetic daily-resolution fixture.

These observations should not be described as production forecasting. They show an early-warning mechanism under a controlled generative pattern, not predictive validity on independently sampled incidents.

# 9. Mechanism Validation

Property tests help distinguish design invariants from performance claims.

### Obsolete-path decay

A historically reinforced legacy recovery edge peaks at influence 5.5 and decays to approximately 0.309 while its historical transition count remains 10 and the edge remains present. The replacement async path reaches influence approximately 3.285 and its recovered terminal ranks above the legacy recovered terminal. This validates the separation between navigational erosion and historical deletion.

### Basin attraction drift

After a recurrent compensation cycle is introduced and repeatedly observed, internal SCC observations increase from 0 to 80 and discovered attractor confidence increases from 0.5 to 1.0. The affected-probe membership gain for the problematic basin is 0.75. These are deterministic fixture-level mechanism checks; they are not estimates of real-world basin dynamics.

### Out-of-order reconciliation

Before the delayed parent arrives, the child has no incoming causal edge. After the parent arrives, exactly one event-provenance causal edge is reconciled, despite the 336-hour event-time gap. This invariant underlies the comparative RQ1 result.

# 10. Threats to Validity

## 10.1 Internal validity

The primary experiments are adversarial but hand-constructed. Their value is mechanistic isolation: each baseline is given a signal that should make it competitive, and the fixture tests whether causal topology contributes distinct information. Hand construction also creates a risk that the score function and fixture co-evolve. We therefore executed the `GlobalFix` reachability-only mutation after the main numerical freeze. Making that 9/10 branch reachable flips the recovery winner to `GlobalFix`, as predicted by the declared gate-plus-Wilson rule. This reduces, but does not eliminate, the risk of an undeclared fixture-specific scoring dependency.

The recovery score contains a fixed coefficient \(\lambda=0.35\). We do not claim this value is optimal. The top-1 ordering in the current fixture is robust to the large Wilson separation between `ProviderFallback` and low-quality reachable distractors, but broader sensitivity analysis remains appropriate.

## 10.2 Construct validity

`causation_id` is treated as authoritative runtime provenance. This is valid only to the extent that the emitting runtime maintains the field correctly. CT-RAG guarantees provenance-preserving retrieval, not the truth of incorrectly emitted provenance.

Likewise, "basin" is an engineered graph construct based on sinks, recurrent SCCs, and bounded reachability. It is not a claim about dynamical systems in the continuous mathematical sense.

The Wilson lower bound is a conservative support score, not a posterior probability and not a causal success probability under intervention.

## 10.3 External validity

The strongest results are controlled synthetic fixtures. They establish that runtime topology can encode useful information under specific adversarial geometries, but they do not establish improvements on production incidents, LongMemEval, AIOps traces, or arbitrary agent tasks.

The repository contains adapters and external-telemetry scaffolding, but a versioned external microservice fixture sufficient for a publication-level external-validity claim is not yet part of the evidence reported here. In telemetry without authoritative causation metadata, CT-RAG deliberately operates with weaker temporal/behavioral relations and should not be described as recovering authoritative causality. Systems that already propagate trace/request context and structured agent/tool telemetry have a substantially lower integration barrier than legacy systems that emit only unstructured logs; retrofitting provenance into the latter remains a real engineering cost, not something the current experiments eliminate.

## 10.4 Statistical validity

The recovery and causal-gap experiments are deterministic mechanism fixtures, not sampled populations; confidence intervals over repeated identical runs would be meaningless. Their evidence comes from adversarial counter-baselines and falsifiable mutation tests rather than frequentist population inference.

The early-degradation experiment explores 28 parameter cells and should be reported as a robustness surface, not as a single tuned threshold. Its current null generator yields FPR 0 for both detectors across the grid. This means the zero FPR gap does **not** provide meaningful matched-operating-point evidence: both methods sit at the floor. The present +1-day claim is therefore supported by same-logic detector comparison plus parameter/phase robustness, not by demonstrated equivalence at non-zero FPR. Before submission, a stochastic null regime with enough independent opportunities to resolve small but non-zero false-positive rates is required.

A separate 2×2 `Evidence Shape Router × CT-RAG` factorial study is preregistered with N=240 held-out paired queries, paired nonparametric bootstrap, Holm FWER correction, and fixed context-token budget. That study has **not** been run and is not treated as evidence in this paper.

# 11. Limitations

First, the current recovery branch-success statistic is computed at the final causal branch before the terminal. More complex recovery trajectories may require state-conditioned success estimation over longer path prefixes, hierarchical context, or covariates. Second, false-rescue rises rapidly for k>1 in the current fixture, so the method presently supports a stronger claim about first-choice discrimination than about high-purity recovery sets. Third, causal reachability depends on graph completeness: a missing causal edge can incorrectly exclude a genuinely feasible route. Fourth, execution-local temporal traversal is fail-safe by default, but broader multi-scope traversal remains an area where consistency windows and explicit cross-scope policies require further validation. Fifth, CT-RAG's terrain uses simple reinforcement and exponential decay; learned or context-dependent terrain dynamics may perform differently.

Finally, CT-RAG should not be described as causal inference. It can preserve and exploit explicit causal provenance and can retrieve observationally different historical trajectories. It cannot infer that taking a retrieved historical action now would cause the same outcome without an identification strategy, intervention evidence, or a valid causal model.

# 12. Discussion

The two primary experiments illustrate complementary roles for topology.

In the causal-gap experiment, topology provides a **relation of authority**. The parent is deliberately poor under both semantic and recency geometry, yet explicit runtime provenance makes it the correct explanation. In the eroded-path experiment, topology provides a **relation of feasibility**. The globally most successful route is deliberately made unavailable from the current state; success statistics alone cannot represent this constraint.

This distinction helps position CT-RAG relative to graph-memory work. A graph is not valuable merely because it is a graph. It is valuable when its edges encode relations that matter to the retrieval question. HippoRAG shows the value of graph-mediated multi-hop association; Graphiti/Zep shows the value of temporal knowledge integration; Topo-RAG shows the value of preserving intrinsic document structure. CT-RAG's strongest evidence concerns a different relation: **runtime experiential topology**, especially authoritative causation and state-conditioned reachability.

The terrain adds a second separation. Historical truth and current navigation need not share one scalar weight. Reinforcement and erosion can adapt what is easy to reach now, while recovery mode can override low navigational influence when preserved historical evidence remains relevant. This separation prevents a self-confirming feedback loop in which only frequently traversed paths remain retrievable and every successful but disused route disappears from practical memory.

The resulting design can be summarized as:

> **Retrieval becomes navigation; learning becomes terrain modification.**

The phrase is architectural rather than cognitive. Retrieval navigates a typed experiential graph; learning changes a non-authoritative overlay. Historical evidence is changed only by new evidence, not by retrieval preference.

# 13. Reproducibility and Artifact Status

The implementation and benchmarks used for this draft are available in the `suissa/Causal-TopologicalRAG` repository. The results reported here correspond to:

- branch: `review/formalization-hardening`;
- commit: `d2c73de6ae1a691ad7d61c9ece26a4e84c93c44c`;
- GitHub Actions CI run: `#210`;
- CI matrix: Python 3.11, 3.12, 3.13, all successful;
- paper values extracted from artifact `benchmarks-python-3.13`.

A subsequent falsification-oriented isolation check changed only `GlobalFix` reachability and passed in CI **#213**, head commit **`9ed7878813b951ca04428e27868d5a4427e8011f`**, across Python 3.11--3.13. It validates the interpretation of reachability as an eligibility gate but does not replace the CI #210 numerical freeze.

Primary artifact files are:

- `eroded-path-rescue/eroded-path-rescue.json`;
- `temporal-terrain-scenarios/temporal-terrain-scenarios.json`;
- `early-degradation-robustness/robustness.json`.

The repository also contains regression, train/dev holdout, terrain-dynamics, and observability artifacts. We intentionally do not use an unrun preregistration or an unavailable external fixture as a reported result.

# 14. Conclusion

CT-RAG treats long-lived agent memory as structured execution experience rather than a flat collection of semantically retrievable records. Its core design separates causal, temporal, and behavioral relations; requires provenance for causal authority; scopes temporal traversal; and overlays a dynamic terrain whose influence can change without rewriting history.

The strongest current evidence comes from two adversarial controlled experiments. First, with 25 semantically close distractors and a 336-hour gap, semantic-only and recency-only retrieval rank the true parent 26th of 26 candidates while runtime causal provenance ranks it first. Second, a globally superior 9/10 recovery is rejected because it is unreachable from the current state, while a reachable 8/10 recovery is ranked first using causal reachability and conservative Wilson-supported historical evidence. These results do not establish universal superiority over vector or graph RAG. They establish a more precise proposition: **there are agent-memory queries for which runtime causal provenance and reachability encode information absent from similarity, recency, and global success statistics.**

The immediate next step is not to expand the claim surface but to test it on versioned external traces and larger held-out recovery sets, and to evaluate whether the same advantage persists under incomplete provenance and stochastic null regimes that produce measurable non-zero false-positive rates.

# References

::: {#refs}
:::