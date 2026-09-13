# Blinded Human Annotation Protocol

Issue #31 requires independent human judgments for categories that event structure alone cannot establish.

## Blinding

Annotators must not see system names, arm identifiers, scores or whether a candidate came from CT-RAG, BM25, dense, GraphRAG or BasinRAG. `python -m ctrag.benchmarks.human_annotation sample ...` generates stable opaque `item_id` values and randomized presentation order from a frozen seed.

## Labels

Each item is scored independently on:

- `relevance`: 0 irrelevant, 1 partially relevant, 2 directly relevant;
- `causal_support`: 0 unsupported, 1 plausible/indirect support, 2 explicit support in the evidence;
- `recovery_usefulness`: 0 not useful, 1 potentially useful, 2 directly useful for recovery;
- `unsupported_claim`: 0 no unsupported claim, 1 contains/encourages an unsupported causal claim.

Annotators must judge only the evidence visible in the item. Temporal adjacency alone is never sufficient for `causal_support=2`.

## Minimum evidence rule

At least two distinct human annotators are required per sampled item before #31 may close. Their raw anonymized judgments must be preserved before adjudication.

Agreement is computed before adjudication with Cohen's kappa for the first two independent annotators per item. Low agreement must be reported and the affected category must not be promoted to high-confidence ground truth.

## Adjudication

Disagreements are not overwritten. A separate adjudication record references `item_id`, both original judgments, the adjudicated value, and a short rationale. Raw annotations remain immutable.

## Machine-readable schema

Each annotation row contains:

```json
{
  "item_id": "opaque-id",
  "annotator_id": "anonymous-stable-id",
  "relevance": 0,
  "causal_support": 0,
  "recovery_usefulness": 0,
  "unsupported_claim": 0
}
```

The annotation file can be joined to experiment results through `item_id`/candidate metadata without exposing the retrieval arm to annotators.

## Current status

The tooling, schema, deterministic sampling and agreement calculation are implemented. Issue #31 must remain open until at least two real human annotators have produced judgments on the frozen sample.
