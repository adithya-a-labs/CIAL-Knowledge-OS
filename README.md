
# CIAL Knowledge OS

An AI-native, fully on-premise enterprise knowledge platform designed for Cochin International Airport Ltd. It enables secure document ingestion, semantic search, agentic RAG, evidence-backed answers, verifier workflows, and department-aware knowledge discovery.

## Vision

CIAL Knowledge OS is designed to become the internal intelligence layer for organizational knowledge: policies, SOPs, project documents, maintenance records, technical manuals, department knowledge, circulars, reports, and operational references.

The goal is not just document chat. The goal is a trusted knowledge operating system where every answer is traceable, verified, permission-aware, and grounded in approved internal sources.

## Core Principles

- Fully local / on-premise deployment
- Open-source LLM support
- Model-agnostic AI through a unified local inference abstraction
- Role-based and department-based access control
- Evidence-backed answers with citations
- Agentic RAG with planner, retriever, critic, verifier, and response agents
- Private employee workspace with isolated personal document storage
- Central repository for organization-approved knowledge
- Auditability, observability, and reliability first

## Phase 1 Scope

- Secure document upload and ingestion
- Chunking, embeddings, and vector search
- Central knowledge repository
- Employee private workspace
- Chat with documents
- Source citations and evidence panel
- Agentic RAG workflow
- Verifier and critic agents
- Admin dashboard
- Department-aware access control
- Local model deployment plan
- Basic analytics and audit logs

## Proposed Stack

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
`src/cial_knowledge_os`.

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
uses `BAAI/bge-m3` and `qwen2.5:7b` by default. No model or document is sent to a
hosted service.

Place non-sensitive text fixtures in `data/sample/`, temporary text input in
`data/raw/`, and approved local PDFs in `data/pdf/`. PDF ingestion prefers Docling
and falls back to PyMuPDF. Each experiment's Qdrant data is written beneath
`data/qdrant/` and must not be committed.

## Experiment Architecture

`notebooks/01_Basic_RAG.ipynb` is the learning and orchestration layer. Reusable
configuration, loading, chunking, embedding, vector storage, retrieval, local
generation, benchmarking, and visualization live in `src/cial_knowledge_os`.
`BasicRAGPipeline` composes those modules while exposing every intermediate result.

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

## Project Rules

All development must follow the rules in:

- `docs/PROJECT_REQUIREMENTS.md` (single source of truth)
- `docs/PROJECT_RULES.md`
- `docs/NOTEBOOK_GUIDELINES.md`

These rules prioritize on-prem deployment, open-source local models, token efficiency, metadata-aware retrieval, citation grounding, and enterprise-grade reliability.
