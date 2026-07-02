
# CIAL Knowledge OS

An enterprise-grade, fully offline, notebook-first RAG platform for enterprise
documentation. The current repository provides completed dense-retrieval Phase
1 and query/context-construction Phase 2 baselines plus reusable local modules.
Agentic workflows, access control, and production applications remain target
capabilities, not current implementation claims.

## Vision

CIAL Knowledge OS is designed to become the internal intelligence layer for organizational knowledge: policies, SOPs, project documents, maintenance records, technical manuals, department knowledge, circulars, reports, and operational references.

The goal is not just document chat. The goal is a trusted knowledge operating system where every answer is traceable, verified, permission-aware, and grounded in approved internal sources.

## Core Principles

- Offline-first, organization-controlled operation
- Open-source and open-weight local models
- No cloud inference or cloud vector database
- Notebook-first experimentation with reusable source modules
- Configuration-driven and reproducible behavior
- Model-agnostic component boundaries
- Token-efficient evidence construction
- Evidence-backed answers, citations, and safe failure
- Auditability, observability, and enterprise readiness

## Completed Baselines

Phase 1 implements PDF and text loading, chunking, local embeddings, embedded
Qdrant, dense retrieval, local Ollama generation, grounded prompts, citations,
basic benchmarking, visualizations, and versioned batch CSV export.

Phase 2 adds deterministic query rewrite, keyword expansion, domain
reformulation, multi-query dense retrieval, evidence collection and exact
deduplication, neighbor expansion, overlap merging, character-bounded context
compression, stronger safe failure, final-evidence citation mapping, retrieval
diagnostics, automated offline evaluation, and regression tests.

The current query rewrite is deterministic, not LLM-based. The current retrieval
system is dense-only and context budgets are character-based.

## Target Production Stack

### Frontend
- React / Next.js
- Tailwind CSS
- Modular dashboard layout

### Backend
- FastAPI or Node.js/NestJS
- PostgreSQL
- pgvector / Qdrant / Milvus
- Redis for job queues and caching

### AI Layer
- Local OSS LLMs
- Llama / Qwen / Mixtral-class models depending on available GPUs
- SentenceTransformers / BGE / E5 embeddings
- Reranker model for retrieval quality

### Infrastructure
- Docker Compose for development
- On-premise GPU workstation deployment
- No AWS or external cloud dependency


## Current Status

Notebook-first RAG experimentation with reusable implementation modules under
`src/cial_knowledge_os`. Phase 1 and Phase 2 notebooks are frozen baselines.
Phase 3 is planned but not implemented; it will focus on BM25, hybrid retrieval,
Reciprocal Rank Fusion, tokenizer-aware context budgets, clickable citations,
and per-run CSV/XLSX/standalone-HTML artifact bundles.

See `docs/CURRENT_STATE.md` for the audited architecture, limitations, output
contracts, frozen notebook policy, and Phase 3 roadmap.

## Local Setup

Python 3.11 or newer is required. Install the pinned local stack and the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

On an isolated host, stage approved wheels first and replace the first command with
`python -m pip install --no-index --find-links <wheelhouse> -r requirements.txt`.

The embedding model must already exist in the local Hugging Face cache, and the
configured Ollama model must already exist in the local Ollama store. The pipeline
uses `BAAI/bge-m3` and `gemma3:12b` by default. No model or document is sent to a
hosted service.

Place non-sensitive text fixtures in `data/sample/`, temporary text input in
`data/raw/`, and approved local PDFs in `data/pdf/`. PDF ingestion prefers Docling
and falls back to PyMuPDF. Each experiment's Qdrant data is written beneath
`data/qdrant/` and must not be committed.

Existing files under `data/sample/` are loaded normally, but the pipeline never
creates synthetic sample documents by default. Demonstration fixtures require an
explicit opt-in:

```python
from cial_knowledge_os import KnowledgeOSConfig, create_sample_airport_documents

config = KnowledgeOSConfig(create_sample_documents=True)  # pipeline.load() opt-in
# Or create them explicitly without changing pipeline configuration:
create_sample_airport_documents(config)
```

## Experiment Architecture

`notebooks/01_Basic_RAG.ipynb` is the learning and orchestration layer. Reusable
configuration, loading, chunking, embedding, vector storage, retrieval, local
generation, benchmarking, and visualization live in `src/cial_knowledge_os`.
`BasicRAGPipeline` composes those modules while exposing every intermediate result.

`notebooks/02_Query_Transformations_and_Context_Construction.ipynb` is the Phase 2
experiment. `Phase2RAGPipeline` extends the basic pipeline with deterministic query
variants, configurable top-10 multi-query retrieval, `(source, page, chunk_id)`
deduplication, neighbor expansion, overlap merging, bounded context construction,
and metadata-rich citations. Phase 1 defaults and APIs remain unchanged.

```python
from cial_knowledge_os import Phase2Config, Phase2RAGPipeline

config = Phase2Config(retrieval_top_k=10, neighbor_window=1)
pipeline = Phase2RAGPipeline(config)
response = pipeline.run("What controls apply before electrical maintenance?")
```

Every Phase 2 stage is available in `response["context_stages"]`; future hybrid
retrieval and reranking components can be inserted at the retrieval and
post-retrieval boundaries without changing ingestion or generation.

Reusable Phase 2 debugging helpers in `visualization.py` convert live pipeline
traces into pandas tables and matplotlib plots. They cover query variants,
single- versus multi-query retrieval, deduplication frequency, neighbor
provenance, score strength, source/page concentration, retrieval funnels,
character-based context compression, section balance, citation quality, batch
answer status, retrieval traces, and per-question latency. Notebook 02 only
supplies real pipeline outputs to these helpers.

Embedded Qdrant permits only one process per storage path. Close other notebook
kernels or clients before reopening the same `data/qdrant/` directory.

## Batch QA Exports

Notebook-defined question lists can be evaluated without notebook-side loops or
file handling:

```python
from cial_knowledge_os import export_batch_answers

csv_path = export_batch_answers(pipeline=pipeline, questions=questions)
```

Exports are written locally beneath `outputs/batch_answers/` using versioned,
non-overwriting filenames. See `docs/BATCH_QA_EXPORT.md` for naming options, input
file support, metrics, and the CSV schema.

The same function accepts `Phase2RAGPipeline`. Phase 2 exports retain all Phase 1
columns and append query variants, retrieval-stage counts, context sizes,
semantic answer status, and a concise retrieval trace. Each exported answer runs
through the complete Phase 2 pipeline; retrieval enhancements are not bypassed.

## Project Rules

All development must follow the rules in:

- `docs/PROJECT_REQUIREMENTS.md` (single source of truth)
- `docs/PROJECT_RULES.md`
- `docs/NOTEBOOK_GUIDELINES.md`

These rules prioritize on-prem deployment, open-source local models, token efficiency, metadata-aware retrieval, citation grounding, and enterprise-grade reliability.

## Documentation

- [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md) describes the audited
  implementation state, limitations, and immediate Phase 3 boundary.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) describes the long-term phase-by-phase
  architectural direction without treating planned capabilities as implemented.
