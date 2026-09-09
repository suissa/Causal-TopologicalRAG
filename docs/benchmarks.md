# Benchmark and ablation harness

## Run and reproduce

Install with `pip install -e ".[dev]"`, then:

```bash
python -m ctrag.benchmarks
python -m pytest
```

To configure a run:

```bash
python -m ctrag.benchmarks --seeds 7 42 2024 --ks 1 3 5 10 --traces 4 --dimensions 256 --max-hops 8 --hop-decay 0.7 --output benchmark-results
```

The first command is equivalent to the configurable command above. Each output
directory contains:

| File | Contents |
| --- | --- |
| `config.json` | Seeds, K values, generator version, trace count, embedding dimensions/name, traversal parameters, all effective weights by mode, candidate/anchor/tie policies, token counter, Python version and SHA-256 fingerprints of package sources |
| `datasets.json` | Full input nodes, fixed timestamps, metadata, typed edges, queries and independent ground-truth labels; its SHA-256 is in config |
| `results.json` | Configuration, per-query observations (including ranked node IDs), and grouped summaries |
| `results.csv` | The same observations; retrieved IDs encoded as a JSON array in one cell |
| `summary.csv` | Long-form dataset/mode/baseline/K/metric table with mean, sample standard deviation and applicable count `n` |
| `table.csv` | Wide table with one mean column per metric |

Rerun the command using the saved configuration values and matching source
fingerprints. Repeated runs on the same Python version are byte-identical,
including across different `PYTHONHASHSEED` values. The Python version field
intentionally differs between CI environments. Text source fingerprints normalize
line endings. There are no wall-clock timestamps, external calls or model downloads.
Running again in the same output directory replaces the six report files.

## Controlled retrieval arms

The harness calls `CTRetriever.search(exhaustive=True)` to score every node except
the supplied anchor. It retrieves once at maximum K and evaluates prefixes for
each cutoff. This avoids the default retriever's mixed semantic/lexical candidate
pruning contaminating the graph-only and lexical-only arms. Scores tie-break by
node ID. All arms receive the same explicit known-state anchor; ground-truth
relevance, distances, paths and basin labels are passed only to the evaluator.

| Arm | Semantic | Lexical | Causal | Topological | Temporal | Behavioral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical_only | 0 | 1 | 0 | 0 | 0 | 0 |
| dense_only | 1 | 0 | 0 | 0 | 0 | 0 |
| dense_lexical | .5 | .5 | 0 | 0 | 0 | 0 |
| graph_topology_only | 0 | 0 | 0 | 1 | 0 | 0 |
| dense_causal | .5 | 0 | .5 | 0 | 0 | 0 |
| dense_topological | .5 | 0 | 0 | .5 | 0 | 0 |
| full_ctrag | Mode-specific existing `RetrievalWeights.for_mode` preset, fully persisted in config | | | | | |

The topology component includes causal/behavioral/temporal reachability and shared
basin affinity, as defined by the current retriever. Thus graph/topology-only is
not a causal-edge-free arm. Basin traversal currently has a fixed eight-hop limit,
recorded separately from `--max-hops`. Components with zero weights may still be
computed by the retriever, but do not affect the ranking. This is a quality
benchmark, not a latency or computational-cost comparison.

“Dense” names refer to the current signed feature-hashing vector/cosine proxy.
It has no learned semantic knowledge and shares tokenization with lexical scoring.
Lexical scoring is the existing normalized IDF-weighted query-term overlap, not
BM25. Comparisons against learned dense encoders, BM25 and real corpora remain
future work; this suite must not be presented as that evidence.

## Deterministic datasets

Each dataset/seed has four independent traces by default, with nine nodes per
trace and three queries per trace. Seeds vary service terminology, opaque node IDs
and insertion order; timestamps come from a fixed epoch. Graph structures are
fixed templates, so different seeds are not independent real-world datasets.

* **Failure/recovery:** accepted request → stale lease → failure → release/refresh
  → retry validation → recovered attractor. Queries ask WHY the failure occurred,
  which recovery evidence follows it, and WHAT_NEXT after remediation.
* **Branching:** accepted request → policy decision, splitting into approval →
  success and rejection → failure. Both terminal nodes are attractors. WHY queries
  target each outcome; WHAT_NEXT at the split labels both downstream branches.
* Each trace also contains two semantically similar distractors and a heartbeat.
  One distractor is temporally adjacent to the failure, but is never a causal
  ancestor. Causal edges carry explicit execution provenance.

Gold paths and distances are authored from template positions, never computed
from the retriever. One-hop relevant nodes have grade 2; other relevant path nodes
have grade 1. The known anchor is excluded from relevance. In these initial tasks,
all relevant nodes are causal evidence, so ordinary and causal recall coincide;
the evaluator accepts separate relevance and causal labels for future fixtures.
Known attractors and complete historical traces are part of the input: this is
retrospective evidence retrieval, not prospective outcome prediction. Recovery
queries retrieve observed remediation, not unseen plans or causal interventions.

## Metric definitions (schema version 1)

Let `R` be the first K returned unique IDs, `G` the positive relevance labels,
and `C = R ∪ {anchor}` the available context graph. Report names `mrr` and `ndcg`
are cutoff-specific: both use only the current K prefix. The implicit anchor is
available for paths/reconstruction but consumes no retrieved-context tokens.

| Metric | Definition |
| --- | --- |
| Recall@K | `|R ∩ G| / |G|` |
| Precision@K | `|R ∩ G| / K`; missing slots count as misses even when K exceeds corpus size |
| MRR | Reciprocal rank of the first positive label within K, or 0 if absent; summary is the mean reciprocal rank |
| nDCG | DCG uses gain `2^grade - 1` and discount `log2(rank + 1)`, normalized by ideal DCG at K |
| Causal Recall@K | Fraction of gold causal node IDs present in R |
| Causal Path Recall | Fraction of complete gold paths whose nodes are all in C and whose directed consecutive edges exist as CAUSAL edges; partial paths receive no credit |
| Causal Distance Error | Mean absolute difference between gold anchor-to-evidence hop count and shortest causal distance in the induced context C, traversing incoming edges for WHY and outgoing edges otherwise. Each missing/disconnected gold node costs `|V|`, the dataset node count, rather than being omitted. Lower is better; penalty scale depends on corpus size |
| Trajectory Reconstruction Accuracy | Sort C by timestamp then ID; compute longest common subsequence with the gold chronological trajectory, divided by the larger sequence length. Penalizes omissions, wrong order and extra nodes. This measures a timestamp-based reconstruction adapter, not ranking order |
| Basin Purity | Fraction of R belonging to the query's gold target attractor basin. This is **retrieved target-basin purity**, not global clustering purity. Branch ancestors may belong to both basins |
| Recovery Path Precision | Fraction of R belonging to the labeled remediation path (excluding the failure anchor); complete ordered-path recovery is measured separately by Causal Path Recall |
| Context-token efficiency | Tokens in positively relevant retrieved nodes divided by all retrieved-node tokens, using `ctrag.embedding.tokenize` (Unicode word/hyphen tokens). This is a deterministic proxy, not an LLM tokenizer or full-prompt cost |
| Context tokens | Denominator of context-token efficiency, also reported for interpreting the ratio |

No applicable ground truth produces JSON `null` / an empty CSV cell, and is
excluded from the mean (`n=0` when the whole group is inapplicable). This includes
trajectory/basin metrics for the two-outcome branch query and recovery precision
for non-recovery queries. With applicable labels, empty retrieval has zero recall,
precision and token efficiency; missing causal distances retain their penalty.
An anchor-only reconstruction can receive partial trajectory credit.

Summaries group by dataset, mode, baseline and K and macro-average applicable
query/seed observations. `std` is sample standard deviation across those
observations (0 for one observation), not a confidence interval or standard error.
The wide paper table contains means; consult `summary.csv` for counts and spread.
These metrics quantify retrieved evidence coverage, not the truth of new causal
claims or the correctness of a downstream generated answer.

## Validation

Tests cover hand-calculated ranking and structural metrics, missing/disconnected
evidence, temporal-edge exclusion, chronological reconstruction, recovery direction,
inapplicable values, ground-truth consistency, exhaustive candidates outside the
default pool, baseline score isolation, stable ties, aggregation/CSV round trips,
and byte-identical CLI output across process hash seeds. CI executes all tests and
the full default benchmark on Python 3.11, 3.12 and 3.13 and uploads all six reports.
