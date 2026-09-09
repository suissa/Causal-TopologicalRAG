# Causal path confidence and provenance

CT-RAG treats causal confidence as evidence over directed causal paths, not as graph proximity alone.

## Provenance calibration

Each causal edge contributes its declared confidence and a provenance factor:

| Provenance | Factor |
| --- | ---: |
| execution | 1.00 |
| workflow | 0.95 |
| dependency | 0.90 |
| event | 0.90 |
| inferred | 0.60 |
| hypothesized | 0.35 |

These factors deliberately keep inferred/hypothesized evidence below otherwise equivalent observed execution evidence.

## Path confidence

For one simple causal path, confidence is the product of:

```text
edge.confidence × min(edge.weight, 1) × provenance_factor
```

Only `EdgeKind.CAUSAL` participates. Temporal and behavioral edges never contribute to causal confidence.

`min(edge.weight, 1)` prevents later navigational reinforcement from manufacturing stronger causal evidence than the underlying edge supports.

## Multiple paths

All simple causal paths up to the requested hop budget are considered. Cycles are safe because a node cannot reappear inside one enumerated path.

The path with greatest confidence is retained as the explainable `best_path`. Multiple independent paths are combined with noisy-OR:

```text
aggregate = 1 - product(1 - path_confidence_i)
```

The aggregate is used as the causal retrieval signal while `RetrievalHit.causal_path` keeps the selected path, its edges, evidence references, provenances, best confidence and aggregate confidence.

## Direction and budgets

`WHY` traverses incoming causal edges (ancestors). `WHAT_NEXT` traverses outgoing causal edges (descendants). Other modes can inspect both directions.

`CTRetriever.search()` supports independent budgets:

```python
retriever.search(..., max_hops=4, ancestor_hops=6, descendant_hops=2)
```

This lets diagnostic queries inspect deeper history without forcing forward-looking queries to expand equally far.
