\# CISG Benchmark Dataset v1



\## Purpose



This benchmark is the official evaluation dataset for the CIAL Knowledge OS project.



It is used to evaluate every notebook in the RAG notebook series.



The benchmark is frozen after release.



Future versions must be released as



\- v2

\- v3

\- ...



instead of modifying this version.



\---



\# Dataset



Questions: 200



Ground truth answers: Yes



Expected keywords: Yes



Source page references: Yes



Categories:



\- Factual

\- Definition

\- Procedure

\- Comparison

\- Executive Summary

\- Enterprise

\- Cross-document

\- Unsupported



\---



\# Source Documents



The benchmark was created from the following CISG/CERT-In publications:



\- CISG-2023-01

\- CISG-2024-01

\- CISG-2024-02

\- CISG-2025-01

\- CISG-2025-02

\- CISG-2025-03

\- CISG-2026-01

\- CISG-2026-02

\- CISG-2026-03



No external knowledge was used.



\---



\# Ground Truth



Ground-truth answers were generated using the source documents and manually reviewed.



Expected keywords were created to support automated evaluation.



Unsupported questions intentionally have no valid answer inside the corpus.



\---



\# Versioning



This benchmark is immutable.



Corrections should create



\- cisg\_benchmark\_v2.csv



instead of modifying v1.



\---



\# Evaluation



The benchmark supports:



\- Retrieval evaluation

\- RAG answer evaluation

\- Keyword scoring

\- Safe-failure validation

\- Citation validation

\- Enterprise QA benchmarking



\---



\# Usage



The benchmark is consumed by:



\- Phase 2 automated evaluation

\- Phase 3 Hybrid Retrieval evaluation

\- Phase 4 Reranking evaluation

\- Phase 5 Agentic RAG evaluation

\- Production Knowledge OS regression testing



\---



\# Folder Structure



```

benchmarks/

&#x20;   cisg/

&#x20;       cisg\_questions\_v1.txt

&#x20;       cisg\_benchmark\_v1.csv

&#x20;       benchmark\_metadata.json

&#x20;       README.md

```

