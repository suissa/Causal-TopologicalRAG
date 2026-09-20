# Reference audit for CT-RAG paper v0.1

Audit date: 2026-09-20

This file records manual verification of recent references whose existence or
characterization materially affects the paper's novelty/related-work claims.
The goal is to prevent an unverified or hallucinated citation from entering a
paper centered on provenance.

## Verified recent references

| Citation key | Verified source | Authors checked | Claim used in CT-RAG | Audit status |
|---|---|---|---|---|
| `hu2025memorysurvey` | arXiv:2512.13564 — *Memory in the Age of AI Agents* | Yuyang Hu, Shichun Liu, Yanwei Yue, et al. | Distinguishes agent memory from RAG/context engineering and organizes memory by forms, functions and dynamics; functional taxonomy includes factual, experiential and working memory. | VERIFIED |
| `yang2026graphmemory` | arXiv:2602.05665 — *Graph-based Agent Memory: Taxonomy, Techniques, and Applications* | Chang Yang, Chuang Zhou, Yilin Xiao, et al. | Establishes graph-based agent memory as a broad design family; taxonomy includes knowledge vs experience memory and non-structural vs structural memory, with extraction/storage/retrieval/evolution lifecycle. | VERIFIED |
| `dai2026gsem` | arXiv:2607.19985 — *Coordinating from Memory: Graph-Structured Experience Reuse for Multi-Agent Adaptation in Dynamic Manufacturing* | Chengxiao Dai, Zhanhui Lin, Zhaokun Yan, Youyang Ni, Chenjun Lei, Luyan Zhang | GSEM encodes historical coordination episodes as heterogeneous relational graphs and retrieves structurally similar past episodes for experience-guided adaptation. | VERIFIED |
| `song2026realm` | arXiv:2609.16053 — *Retrieval-Driven Memory Reconsolidation for Long-Term LLM Agents* | Yuanyi Song, Yukai Wang, Xinbei Ma, Zhihui Fu, Jianghao Lin, Weiwen Liu, Jun Wang, Huarong Deng, Yong Yu, Weinan Zhang | REALM organizes memory as a heterogeneous cognitive graph, retrieves through graph-search atoms, and reconsolidates/reorganizes memory using retrieval feedback. | VERIFIED |
| `dantart2026toporag` | arXiv:2601.10215 — *Topo-RAG: Topology-aware Retrieval for Hybrid Text-Table Documents* | Alex Dantart, Marco Kóvacs-Navarro | Preserves intrinsic structure of hybrid text/table artifacts; routes narrative to dense retrieval and tables to cell-aware late interaction rather than flattening everything to text. | VERIFIED |

## Additional recent negative-result cross-check

`rusu2026selective` was also checked against arXiv:2608.28978,
*Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents*
(Theo Rusu, Sourena Khanzadeh, Manar Alalfi). The abstract reports that its
graph pipeline does not outperform a flat vector baseline under the matched
candidate-generation budget, while selective pruning removes memory with little
measured quality loss. The CT-RAG paper's characterization is consistent with
that abstract.

## Canonical identifiers

- https://arxiv.org/abs/2512.13564
- https://arxiv.org/abs/2602.05665
- https://arxiv.org/abs/2607.19985
- https://arxiv.org/abs/2609.16053
- https://arxiv.org/abs/2601.10215
- https://arxiv.org/abs/2608.28978

## Audit discipline

A citation is marked VERIFIED here only when both conditions hold:

1. the paper/identifier and author metadata were located in the primary archive
   or an authoritative publication record; and
2. the specific characterization used by CT-RAG is supported by the paper's
   abstract or text inspected during the audit.

Verification of existence is not treated as verification of every result in the
cited paper. CT-RAG should avoid importing stronger claims than those explicitly
checked here.
