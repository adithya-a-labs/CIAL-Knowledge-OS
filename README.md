
# CIAL Knowledge OS

An AI-native, fully on-premise enterprise knowledge platform designed for Cochin International Airport Ltd. It enables secure document ingestion, semantic search, agentic RAG, evidence-backed answers, verifier workflows, and department-aware knowledge discovery.

## Vision

CIAL Knowledge OS is designed to become the internal intelligence layer for organizational knowledge: policies, SOPs, project documents, maintenance records, technical manuals, department knowledge, circulars, reports, and operational references.

The goal is not just document chat. The goal is a trusted knowledge operating system where every answer is traceable, verified, permission-aware, and grounded in approved internal sources.

## Core Principles

- Fully local / on-premise deployment
- Open-source LLM support
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

Planning and foundation stage for the CIAL Knowledge OS internship/project.

## Project Rules

All development must follow the rules in:

- `docs/PROJECT_RULES.md`
- `docs/NOTEBOOK_GUIDELINES.md`

These rules prioritize on-prem deployment, open-source local models, token efficiency, metadata-aware retrieval, citation grounding, and enterprise-grade reliability.
