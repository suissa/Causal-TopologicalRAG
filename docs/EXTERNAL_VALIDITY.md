# External Validity Protocol

This document defines the first Gate C evaluation path for CT-RAG using execution traces that were not produced by the synthetic benchmark generator.

## Sources

Versioned fixtures live in `research/external/github-actions-v1.json` and currently include four public GitHub Actions jobs:

- `suissa/Causal-TopologicalRAG` science run `34357104431` (successful real project execution);
- `suissa/Causal-TopologicalRAG` science run `34355758910` (failed real project execution);
- `Basinfy/BasinRAG` run `34723586051` (independent public repository, failed CI);
- `psf/requests` run `34660530936` (independent public repository, successful workflow).

The two latter repositories satisfy the independent-public-source requirement for issue #20. Source repository, run ID, job ID, URL, observed conclusion and observed step list are retained in the fixture.

## Transformation rules

A GitHub Actions step becomes one `MemoryNode`. Consecutive observed steps become `TEMPORAL` edges only.

**Sequence is not causation.** No `CAUSAL` edge is created from step order. A future adapter may create `CausalProvenance.WORKFLOW` edges only when the original source declares an explicit dependency such as `needs:` or another provenance-bearing dependency relation.

Where exact per-step timestamps are unavailable, deterministic order-surrogate timestamps are created solely to satisfy the memory schema and reconstruct observed order. They must not be interpreted as measured execution duration.

## Ground truth

Labels are derived from source-observed facts, not from CT-RAG output:

- `WHAT_NEXT`: the next observed step in the job;
- `WHY` track on failures: the immediately preceding observed step, explicitly labeled as predecessor evidence, **not causal explanation**;
- `RECOVERY` track: the first later successful cleanup/continuation step after a failure, if one exists, explicitly labeled observational.

Consequently, causal-specific metrics are intentionally `None` for this first external dataset. This is a feature of the protocol: missing causal evidence remains missing instead of being fabricated.

## Evaluation arms

`python -m ctrag.benchmarks.external_validity` evaluates the same observable corpus with:

- dense proxy retrieval;
- lexical retrieval;
- hybrid RRF retrieval;
- CT-RAG with the true observed anchor and temporal/topological navigation.

Results are reported per dataset. Synthetic and external results must never be pooled into one headline number.

## Scientific interpretation

This benchmark tests whether CT-RAG's navigation model remains useful on real procedural traces when explicit causal metadata is absent. It does **not** test the central causal-edge hypothesis by itself. A negative result here would mean causal/topological advantages do not automatically transfer to generic workflow logs without causal provenance.

The final preregistered synthetic holdout remains sealed throughout Gate C development.
