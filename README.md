<img width="1672" height="941" alt="1000448050" src="https://github.com/user-attachments/assets/604de328-89b4-47cd-8015-8a299ad1c2e8" />

# Causal-Topological RAG (CT-RAG)

CT-RAG is an experimental retrieval architecture for stateful agents and event-driven systems. It combines semantic and lexical retrieval with explicit causal evidence, topological navigation, behavioral traces, dynamic terrain and basins of attraction.

The core question is not only **“what looks like this?”**, but also **“where am I, how did I get here, and what happened the last time this terrain was traversed?”**

## Research artifacts

- [Experimental validation report](REPORT.md)
- [Formal research specification and related work](docs/RESEARCH.md)
- [Reproducibility protocol](docs/REPRODUCIBILITY.md)
- [Topology: construction, meaning and structural parts](docs/TOPOLOGY.md)
- [Storage boundaries and adapter conformance](docs/STORAGE.md)
- [Benchmark methodology and metrics](docs/benchmarks.md)
- [Implementation roadmap](IMPLEMENTATION_PLAN.md)

Additional public contracts:

- [core node/edge model](docs/model-contracts.md);
- [causal path confidence and provenance](docs/causal-paths.md);
- [basins and attractors](docs/basins.md);
- [dense/lexical retrieval adapters and RRF](docs/retrieval-adapters.md).

## Design principles

- **Causality is explicit.** Temporal adjacency never becomes a causal edge by itself.
- **Causal provenance is first-class.** Execution/workflow/event evidence remains distinguishable from inferred or hypothesized links.
- **Retrieval is navigational.** Semantic/lexical search identifies anchors; CT-RAG then traverses local causal/topological neighborhoods.
- **Basins are structural.** Attractors define queryable reverse-reachable regions and can be manual or empirically discovered.
- **Terrain is non-authoritative.** Reinforcement and erosion change navigational influence without rewriting historical events or causal edges.
- **Storage is replaceable.** Database adapters must preserve causal semantics and ranking inputs.
- **Stateful-agent friendly.** Event-sourced systems can project causation/correlation/execution identifiers directly into the terrain.
- **Reproducibility is executable.** Benchmarks and paper artifacts are generated from machine-readable inputs in CI.

## Architecture

```text
Authoritative Event Source
          |
          v
EventProjector
          |
          v
CausalTopology
  |       |       |
  |       |       +--> basins / attractors
  |       +----------> DynamicTerrain overlay
  +------------------> vector / lexical signals
          |
          v
staged CT-RAG retrieval
  anchor search
    -> basin/topology expansion
    -> directed causal traversal
    -> mode reranking
          |
          v
explainable ranked context
```

The baseline score combines semantic similarity, lexical relevance, causal proximity/confidence, topological proximity, temporal relevance and behavioral affinity. Staged retrieval additionally exposes anchors, traversal stages and mode-specific priors. `TerrainAwareRetriever` can apply empirical path reinforcement as a separate final navigation signal.

Supported query modes:

```text
similar
why
what_next
recovery
counterfactual
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from datetime import datetime, timezone

from ctrag import (
    CausalProvenance,
    Edge,
    EdgeKind,
    MemoryNode,
    QueryMode,
    CausalTopology,
    CTRetriever,
)

topology = CausalTopology()
topology.add_node(MemoryNode(
    id="authorized",
    text="Payment was authorized by the provider",
    timestamp=datetime.now(timezone.utc),
    metadata={"execution_id": "exec-1", "intent_id": "checkout"},
))
topology.add_node(MemoryNode(
    id="stock-error",
    text="Inventory reservation failed because stock changed",
    timestamp=datetime.now(timezone.utc),
    metadata={"execution_id": "exec-1", "intent_id": "checkout"},
))
topology.add_edge(Edge(
    source="authorized",
    target="stock-error",
    kind=EdgeKind.CAUSAL,
    provenance=CausalProvenance.EXECUTION,
    confidence=1.0,
))

retriever = CTRetriever(topology)
result = retriever.why("stock-error", "why did the inventory reservation fail?")

for hit in result.hits:
    print(hit.node.id, round(hit.score, 3), hit.components, hit.causal_path)
```

## Event-sourced ingestion

`EventProjector` converts execution events into memory nodes. Explicit `causation_id` creates a causal edge with evidence; if the parent event arrives after the child, pending causation is reconciled deterministically when the parent appears. Events in the same execution receive temporal/behavioral relations, but those relations are never promoted to causality by order alone.

The projector also supports configurable nested field mappings, canonical event fingerprints, idempotent replay and explicit conflict detection for reused event IDs.

See `examples/stateful_agent.py` and `tests/fixtures/multi_step_trace.ndjson`.

## Retrieval adapters

`CTRetriever` accepts pluggable dense and lexical adapters while retaining dependency-free defaults. Built-ins include deterministic hashing embeddings, IDF overlap, BM25, optional Sentence Transformers, an OpenAI-compatible embeddings endpoint adapter, and deterministic Reciprocal Rank Fusion.

See [`docs/retrieval-adapters.md`](docs/retrieval-adapters.md).

## Dynamic terrain

`DynamicTerrain` tracks observed transition frequency and navigational influence separately from the authoritative graph. Repeated trajectories can be reinforced; erosion decays influence without deleting historical evidence. The terrain can discover structural sink and recurrent-SCC attractors and measure basin drift between snapshots.

`TerrainAwareRetriever` consumes this overlay without changing the historical baseline `CTRetriever.search()` implementation used by the published validation report.

## Persistence

`src/ctrag/storage.py` defines:

```text
TopologyView
MemoryStore
VectorIndex
TopologyStore
EventSource
TerrainStore
```

`SQLiteCTStore` is the persistent local reference adapter. Conformance tests prove that save/reload preserves nodes, embeddings, causal provenance/confidence/evidence, attractors, terrain influence and CT-RAG ranking inputs/results.

See [`docs/STORAGE.md`](docs/STORAGE.md).

## Reproducible benchmarks

Run every baseline and ablation with:

```bash
python -m ctrag.benchmarks
```

The default controlled run evaluates lexical only, dense only, dense+lexical, graph/topology only, dense+causal, dense+topological and full CT-RAG on deterministic failure/recovery and branching traces, including `WHY`, `WHAT_NEXT` and `RECOVERY` queries.

Results in `benchmark-results/` include generated datasets/labels, seeds/configuration, source fingerprints, per-query JSON/CSV, summaries and paper-table CSV.

The historical `REPORT.md` benchmark uses the deterministic `HashingEmbedder` proxy and `IdfOverlapRetriever`, not a trained semantic embedding model or production BM25. Later adapters do not retroactively redefine those reported results.

## Generate paper-ready artifacts

After the benchmark:

```bash
python -m ctrag.research_artifacts \
  --benchmark-dir benchmark-results \
  --out research-artifacts \
  --k 3
```

This generates:

```text
research-artifacts/
  topology.dot
  causal-path.dot
  benchmark-k3.md
  README.md
  manifest.json
```

`manifest.json` fingerprints benchmark inputs and generated outputs with SHA-256. CI runs this command on Python 3.11–3.13 and uploads both benchmark and research-artifact bundles.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Implemented scope

The repository now includes:

- typed/serializable memory nodes and edges;
- explicit causal provenance, confidence and evidence;
- pluggable semantic/lexical retrieval and RRF;
- provenance-aware causal path reconstruction;
- basins, attractor descriptors, boundaries and neighboring basins;
- idempotent Event Sourcing projection with out-of-order causation reconciliation;
- staged `WHY`, `WHAT_NEXT`, `RECOVERY` and `COUNTERFACTUAL` retrieval;
- reproducible seven-arm benchmark/ablation harness;
- dynamic terrain reinforcement, erosion, SCC/sink attractor discovery and basin drift;
- terrain-aware reranking;
- storage protocols plus SQLite persistence/conformance tests;
- deterministic generation of paper-ready DOT figures/tables/manifests;
- experimental validation and formal/reproducibility documentation.

## Related work

CT-RAG builds on the broader RAG line of work, graph-based retrieval and the topological retrieval direction exemplified by BasinRAG. Its distinguishing research target is stateful/event-driven memory where explicit execution provenance can be preserved as causal structure rather than reconstructed solely from textual similarity.

The sourced related-work discussion is in [`docs/RESEARCH.md`](docs/RESEARCH.md), including:

- Lewis et al., RAG (2020);
- Edge et al., GraphRAG (2024);
- Martins, BasinRAG (2026), DOI `10.5281/zenodo.22664948`;
- Fowler, Event Sourcing (2005).

## Status

Research prototype with controlled positive empirical evidence. `REPORT.md` demonstrates the current concept on synthetic event-sourced traces, but does not claim universal superiority over learned dense retrieval, GraphRAG, BasinRAG or real-world causal-inference systems. External-validity experiments remain future research.
