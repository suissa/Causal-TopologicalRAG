# CT-RAG paper

Canonical manuscript sources for the first CT-RAG paper.

- `CT-RAG-paper-v0.1.md` — source manuscript with citation keys.
- `CT-RAG-paper-v0.1.tex` — LaTeX source.
- `ctrag-references.bib` — bibliography.

## Evidence freeze for v0.1

The numerical results in this draft are intentionally tied to the pre-paper evidence freeze:

- branch: `review/formalization-hardening`
- evidence commit: `d2c73de6ae1a691ad7d61c9ece26a4e84c93c44c`
- GitHub Actions run: `#210`
- CI: Python 3.11 / 3.12 / 3.13 successful
- benchmark artifact used for manuscript values: `benchmarks-python-3.13`

Subsequent implementation commits must not silently rewrite these reported numbers. New experiments should update the manuscript only with a new explicit evidence freeze.

## Claim discipline

The paper distinguishes runtime causal provenance from causal inference. Controlled synthetic mechanism experiments are not presented as production generalization. The preregistered Evidence Shape Router × CT-RAG factorial study is not a reported result until executed.
