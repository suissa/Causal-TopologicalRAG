# CCTDb behavioral memory contract

## Epistemic boundary

CTEG remains the authoritative record of observed experience. The cognitive
memory layer stores interpretations, baselines, and snapshots. It references
CTEG evidence but never writes inferred beliefs into CTEG and never turns
topological adjacency, event order, or trace parent/child structure into a
causal edge.

## Implemented contract

`ctrag.cognitive_memory` defines the 15 requested node kinds and typed
relations, explicit evidence references, confidence-bounded hypotheses,
validity intervals, source signal/transition/pattern references, inference
method and model/skill versions, and BehaviorID-LT snapshots.

`CognitiveMemoryStore` is the persistence protocol. The included
`SQLiteCognitiveMemoryStore` is a runnable reference adapter and test
implementation; it is **not** claimed to be the production CCTDb backend.
A CCTDb integration can implement the same protocol without changing the
domain contract.

## Retrieval

- `get_baselines(subject_id)` returns only that subject's prosody/text baselines.
- `get_hypotheses(subject_id, valid_at=...)` applies the hypothesis validity window.
- `get_hypothesis_evidence(id)` returns support and counter-evidence separately.
- `get_snapshot(id)` restores BehaviorID and hypothesis references plus baseline refs.

Every hypothesis records confidence, support, counter-evidence, temporal
validity, signal/transition/pattern refs, inference method, model version and
skill version. No fixed personality labels are introduced.

## Example

```python
from datetime import datetime, timezone
from ctrag.cognitive_memory import SQLiteCognitiveMemoryStore

store = SQLiteCognitiveMemoryStore("behavior-memory.sqlite")
snapshot = store.get_snapshot("snapshot-2026-11-08-subject-42")
hypotheses = store.get_hypotheses(
    "subject-42", valid_at=datetime.now(timezone.utc)
)
```

## Limitations

The reference adapter uses SQLite and JSON payloads for local reproducibility.
It does not provide a native CCTDb driver, encryption/key management, multi-user
authorization, retention policy, or production-scale graph indexing. Those are
deployment concerns and must be supplied by a concrete CCTDb backend.
