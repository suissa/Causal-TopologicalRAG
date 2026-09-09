# Retrieval adapters

CT-RAG separates retrieval semantics from the concrete dense and lexical implementations.

## `EmbeddingProvider`

Any dense adapter must implement:

```python
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...
```

Built-ins:

- `HashingEmbedder`: deterministic, dependency-free research fallback used by the original synthetic report;
- `SentenceTransformersEmbedder`: optional local model adapter loaded lazily from `sentence-transformers`;
- `OpenAICompatibleEmbedder`: stdlib HTTP adapter for an OpenAI-compatible `/embeddings` endpoint.

No external model or network call is required by the unit test suite.

## `LexicalRetriever`

Lexical adapters implement:

```python
class LexicalRetriever(Protocol):
    def score(self, query: str, documents: Mapping[str, str]) -> dict[str, float]: ...
```

Built-ins:

- `IdfOverlapRetriever`: preserves the dependency-free lexical scoring used by the first benchmark/report;
- `BM25Retriever`: dependency-free Okapi BM25 with corpus-local score normalization.

`CTRetriever(..., lexical_retriever=...)` accepts either implementation.

## Independent ranking APIs

The retriever exposes independent ranking surfaces so dense, lexical and hybrid behavior can be benchmarked without topology:

```python
retriever.rank_dense(query, k=10)
retriever.rank_lexical(query, k=10)
retriever.rank_hybrid_rrf(query, k=10)
```

The full `search()` API continues to combine these retrieval components with causal/topological/temporal/behavioral signals.

## Reciprocal Rank Fusion

`reciprocal_rank_fusion()` combines any number of ranked ID sequences. For a document at rank `r`, each ranking contributes:

```text
1 / (rank_constant + r)
```

Duplicate IDs inside one ranking count only at their first position. Ties are resolved by document ID, so fixed input rankings produce deterministic output.

## Reproducibility note

The existing `REPORT.md` remains tied to `HashingEmbedder + IdfOverlapRetriever`. Adding BM25 or learned embeddings does not retroactively change those results. New experiments must persist the adapter/model names and settings alongside benchmark results.
