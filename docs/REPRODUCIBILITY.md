# CT-RAG Reproducibility Protocol

This document defines the commands and artifacts required to reproduce the current controlled CT-RAG experiment.

## Environment

Supported CI interpreters:

```text
Python 3.11
Python 3.12
Python 3.13
```

The default controlled benchmark requires no remote model and no API key. It intentionally uses the deterministic local `HashingEmbedder` baseline.

## 1. Fresh clone

```bash
git clone https://github.com/suissa/Causal-TopologicalRAG.git
cd Causal-TopologicalRAG
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell, activate the virtual environment with the corresponding `.venv\\Scripts\\Activate.ps1` command.

## 2. Run the test suite

```bash
pytest
```

Tests cover model invariants, retrieval adapters, causal paths, basins/attractors, Event Sourcing ingestion, query modes, benchmark metrics, dynamic terrain, SQLite conformance and research-artifact generation.

## 3. Reproduce benchmark data

```bash
python -m ctrag.benchmarks
```

Default output directory:

```text
benchmark-results/
```

Expected machine-readable files:

```text
config.json
datasets.json
results.json
results.csv
summary.csv
table.csv
```

`config.json` persists:

- seeds;
- K values;
- trace count;
- embedding dimensionality;
- hop configuration;
- effective retrieval weights;
- generator version;
- dataset fingerprint;
- Python runtime version;
- source-file SHA-256 fingerprints.

`datasets.json` persists the generated nodes, edges, query labels and ground-truth causal paths used by that run.

## 4. Generate paper-ready artifacts

```bash
python -m ctrag.research_artifacts \
  --benchmark-dir benchmark-results \
  --out research-artifacts \
  --k 3
```

Generated files:

```text
research-artifacts/
  topology.dot
  causal-path.dot
  benchmark-k3.md
  README.md
  manifest.json
```

`manifest.json` records:

- the exact benchmark command;
- the exact artifact command;
- embedded benchmark configuration;
- SHA-256 of benchmark inputs;
- SHA-256 of generated outputs.

The generated benchmark table is derived from `benchmark-results/table.csv`; reported values are not manually copied into the generator.

## 5. Render figures

The repository stores Graphviz DOT as the canonical figure source so generation has no Graphviz runtime dependency.

If Graphviz is installed:

```bash
dot -Tsvg research-artifacts/topology.dot -o research-artifacts/topology.svg
dot -Tsvg research-artifacts/causal-path.dot -o research-artifacts/causal-path.svg
```

Equivalent DOT-compatible renderers may be used. Rendering should change presentation only, not graph contents.

## 6. CI reproduction

`.github/workflows/ci.yml` executes, for every supported Python version:

```text
install
  -> pytest
  -> python -m ctrag.benchmarks
  -> python -m ctrag.research_artifacts
  -> upload benchmark artifacts
  -> upload research artifacts
```

CI therefore validates both the experiment and the transformation from machine-readable results to research artifacts.

## 7. Reproducing the published validation report

[`REPORT.md`](../REPORT.md) records the controlled validation performed on benchmark commit:

```text
f4827e223802b1c10cd7d5c268b00da561526882
```

For exact historical reproduction, check out that commit and run the benchmark command described above. Later commits add capabilities but deliberately preserve the historical `search()` benchmark path rather than silently redefining the original reported experiment.

## 8. Interpretation constraints

A reproduced result demonstrates deterministic behavior of the current controlled experiment. It does not by itself establish external validity.

In particular, the current published report does not establish superiority over:

- production learned dense retrievers;
- a tuned BM25 baseline on shared real-world corpora;
- GraphRAG on a shared task;
- BasinRAG on a shared task;
- causal-inference methods with interventional ground truth.

Those require separate experiment manifests and shared evaluation conditions.

## 9. Causal evidence policy

Reproduction must preserve these semantics:

```text
TEMPORAL != CAUSAL
BEHAVIORAL != CAUSAL
observed causal provenance != inferred provenance
historical divergence != proven counterfactual effect
```

Changing a storage adapter, vector index, graph store or figure renderer is valid only if the corresponding conformance checks demonstrate that causal semantics and ranking inputs are preserved.
