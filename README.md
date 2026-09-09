<img width="1672" height="941" alt="1000448050" src="https://github.com/user-attachments/assets/604de328-89b4-47cd-8015-8a299ad1c2e8" />


# Causal-Topological RAG (CT-RAG)

CT-RAG is an experimental retrieval architecture for stateful agents and event-driven systems. It combines semantic and lexical retrieval with explicit causal evidence, topological navigation, behavioral traces, and basins of attraction.

The core question is not only **“what looks like this?”**, but also **“where am I, how did I get here, and what happened the last time this terrain was traversed?”**

## Design principles

- **Causality is explicit.** Temporal adjacency never becomes a causal edge by itself.
- **Causal provenance is first-class.** Execution/workflow evidence is distinguishable from inferred or hypothesized links.
- **Retrieval is navigational.** Semantic search identifies anchors; CT-RAG then traverses local causal/topological neighborhoods.
- **Basins are structural.** Attractors can define reverse-reachable regions of the execution topology.
- **Stateful-agent friendly.** Event-sourced systems can project causation/correlation/execution identifiers directly into the terrain.
- **Dependency-light MVP.** The core implementation runs on the Python standard library; `pytest` is optional for tests.

Public contracts:

- [topology: construction, meaning and structural parts](docs/TOPOLOGY.md);
- [core node/edge model](docs/model-contracts.md);
- [causal path confidence and provenance](docs/causal-paths.md);
- [basins and attractors](docs/basins.md);
- [dense/lexical retrieval adapters and RRF](docs/retrieval-adapters.md).

## Architecture

```text
query
  |
  v
semantic + lexical anchor search
  |
  v
anchor states
  |
  +--> causal neighborhood (past/future)
  +--> behavioral neighborhood
  +--> basin membership / attractors
  |
  v
multi-signal scoring
  |
  v
ranked contextual memories
```

The score combines semantic similarity, lexical relevance, causal proximity/confidence, topological proximity, temporal relevance and behavioral affinity. Weights change according to query mode (`similar`, `why`, `what_next`, `recovery`, `counterfactual`).

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
hits = retriever.search(
    "why did the inventory reservation fail?",
    mode=QueryMode.WHY,
    anchor_ids=["stock-error"],
)
for hit in hits:
    print(hit.node.id, round(hit.score, 3), hit.components)
```

## Event-sourced ingestion

`EventProjector` converts execution events into memory nodes. Explicit `causation_id` creates a causal edge only when the referenced event exists. Events in the same execution receive temporal/behavioral relations, but those relations are not promoted to causality.

See `examples/stateful_agent.py`.

## Retrieval adapters

`CTRetriever` accepts pluggable dense and lexical adapters while retaining dependency-free defaults. Built-ins include deterministic hashing embeddings, IDF overlap, BM25, optional Sentence Transformers, an OpenAI-compatible embeddings endpoint adapter, and deterministic Reciprocal Rank Fusion. See [`docs/retrieval-adapters.md`](docs/retrieval-adapters.md).

## Reproducible benchmarks

After installation, run every baseline and ablation with one command:

```bash
python -m ctrag.benchmarks
```

This runs lexical only, dense only, dense+lexical, graph/topology only, dense+causal, dense+topological and full CT-RAG on deterministic failure/recovery and branching success/failure traces, including explicit `WHY` queries. The default run produces 2,016 query/K/baseline observations across three seeds.

Results in `benchmark-results/` include the complete generated datasets and labels, seeds/configuration, source fingerprints, per-query JSON/CSV, a summary with applicable sample counts and standard deviations, and a wide `table.csv` for paper tables. CI runs the same command on Python 3.11–3.13 and uploads the outputs as artifacts.

All seven historical report arms use the same known anchor and exhaustive candidate corpus. In `REPORT.md`, **dense means the deterministic HashingEmbedder proxy and lexical means IdfOverlapRetriever**, not a trained semantic embedding model or BM25. Adding adapters does not retroactively change the report.

See [benchmark methodology and metric definitions](docs/benchmarks.md).

## Implemented prototype

Implemented so far:

- memory nodes and typed edges;
- causal provenance, evidence and confidence;
- deterministic model serialization contracts;
- pluggable dense/lexical retrieval adapters;
- deterministic hashing embeddings and IDF overlap fallback;
- BM25 and RRF;
- optional Sentence Transformers/OpenAI-compatible embedding adapters;
- directed causal/topological traversal;
- explicit attractors and basins of attraction;
- query-mode-dependent scoring;
- Event Sourcing projection;
- reproducible benchmark harness and validation report.

The issue-by-issue roadmap is in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Related work

CT-RAG is directly motivated by the topological retrieval intuition demonstrated by **BasinRAG**, which uses functional-graph topology and dynamical basins for retrieval. CT-RAG extends the idea toward stateful systems where part of the topology can come from observed execution causality rather than document adjacency alone.

- BasinRAG: https://github.com/Basinfy/BasinRAG

## Status

Research prototype. The current code is intended to make the CT-RAG hypothesis executable and benchmarkable; it is not yet a production RAG framework.
