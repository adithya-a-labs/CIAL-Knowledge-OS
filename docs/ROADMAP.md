# CIAL Knowledge OS Roadmap

## Purpose of This Roadmap

This roadmap describes the long-term, phase-by-phase evolution of CIAL Knowledge
OS from an offline RAG notebook into an enterprise-grade, on-premises Knowledge
OS.

[`CURRENT_STATE.md`](CURRENT_STATE.md) records what is implemented and true
today. This file describes the intended direction. Planned capabilities in this
roadmap are not current implementation claims, and each phase remains subject to
benchmark evidence and documented design review.

## Project Vision

CIAL Knowledge OS is intended to become an enterprise-grade, fully offline,
notebook-first retrieval-augmented generation platform for secure enterprise
documentation. It should remain model-agnostic, evidence-grounded, and
citation-driven while evolving toward production-ready deployment on
organization-controlled infrastructure.

The current repository provides completed Phase 1 and Phase 2 experimental
baselines and an implemented Phase 3 hybrid architecture awaiting full local
benchmark qualification. Reranking, agentic workflows, enterprise access
controls, and a production interface remain planned.

## Guiding Principles

- **Offline-first:** Core workflows must operate without internet access.
- **No cloud inference:** Documents, prompts, embeddings, and answers remain on
  organization-controlled infrastructure.
- **No cloud vector databases:** Retrieval storage must be local or self-hosted.
- **Open-source and open-weight preference:** Models and supporting components
  should be locally deployable and replaceable.
- **Notebook-first experimentation:** New techniques are first made inspectable
  and measurable in phase-specific notebooks.
- **Frozen phase baselines:** Completed notebooks remain unchanged and runnable.
- **Reusable production modules:** Reusable behavior belongs under
  `src/cial_knowledge_os/`, not in large notebook cells.
- **Configuration-driven design:** Paths, models, retrieval modes, budgets,
  output locations, and filenames must not be scattered as hardcoded literals.
- **Backward compatibility:** New internal architecture should preserve existing
  external contracts.
- **Reproducibility:** Every comparison should record effective configuration,
  metrics, and artifacts.
- **Evidence before generation:** Retrieval and evidence inspection precede
  answer generation.
- **Enterprise readiness:** Privacy, traceability, reliability, and eventual
  authorization requirements shape architectural choices.
- **Modularity:** Ingestion, retrieval, ranking, context construction,
  generation, citation, evaluation, and reporting remain replaceable stages.
- **Token efficiency:** Improve evidence quality and context selection before
  increasing model or context size.
- **No hardcoded paths:** Repository and deployment paths must resolve through
  configuration.

## Architecture Evolution Overview

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Basic Offline RAG | ✅ Completed |
| Phase 2 | Query Transformations and Context Construction | ✅ Completed |
| Phase 3 | Hybrid Retrieval | 🧪 Implemented / Qualification Pending |
| Phase 4 | Reranking and Precision Optimization | 🔮 Planned |
| Phase 5 | Agentic RAG and Multi-step Retrieval | 🔮 Planned |
| Phase 6 | Production Hardening and Enterprise UX | 🔮 Planned |
| Phase 7 | Enterprise Knowledge OS | 🔮 Long-term Target |

## Detailed Phase Breakdown

### Phase 1 — Basic Offline RAG

**Status:** Completed and frozen.

**Objective**

Establish a transparent, fully local dense-retrieval RAG baseline.

**Why this phase exists**

A simple baseline makes later retrieval and orchestration improvements
measurable. It also validates that document processing, local retrieval, local
generation, and citations can work without cloud dependencies.

**Major capabilities**

- PDF and text loading;
- metadata-preserving chunking;
- local embeddings;
- embedded local Qdrant storage;
- dense semantic retrieval;
- local Ollama generation;
- grounded prompting and insufficient-evidence instructions;
- source, page, chunk, score, and metadata-aware citations;
- basic benchmarking and visualization; and
- modular source structure under `src/cial_knowledge_os/`.

**Expected deliverables**

- Frozen `notebooks/01_Basic_RAG.ipynb`.
- Reusable `BasicRAGPipeline` and supporting modules.
- Versioned batch answer CSV exports.
- Inspectable retrieval, generation, citation, and latency outputs.

**Exit criteria**

Met: the local pipeline is modular, testable, inspectable, citation-aware, and
usable as the frozen dense top-k baseline for Phase 2.

### Phase 2 — Query Transformations and Context Construction

**Status:** Completed and frozen.

**Objective**

Improve dense evidence discovery and context quality without replacing the
Phase 1 contracts.

**Why this phase exists**

One user query and one top-k list can miss relevant language or produce
fragmented context. Phase 2 tests whether deterministic query variants and
explicit evidence post-processing improve the baseline while keeping every
stage inspectable.

**Major capabilities**

- original, deterministically rewritten, keyword-expanded, and
  domain-reformulated queries;
- configurable multi-query dense retrieval;
- retrieval fusion through combined evidence collection;
- deduplication by `(source, page, chunk_id)`;
- source-relative neighbor expansion;
- overlap merging;
- context construction and compression with a character-based budget;
- safe insufficient-evidence behavior;
- citation mapping to final retained evidence;
- retrieval and context visualizations;
- deterministic automated evaluation;
- a self-contained offline evaluation dashboard; and
- unit and regression tests.

The completed rewrite is deterministic and does not use an LLM. Token values are
estimated for reporting; the enforced budget is character-based.

**Expected deliverables**

- Frozen `notebooks/02_Query_Transformations_and_Context_Construction.ipynb`.
- Frozen Phase 2 automated-evaluation notebook.
- `Phase2RAGPipeline`, query transformation, context construction, retrieval
  post-processing, evaluation, dashboard, and visualization modules.
- Phase 2 batch and evaluation artifacts.
- Frozen CISG benchmark compatibility.

**Exit criteria**

Met: Phase 2 runs through reusable modules, preserves Phase 1 behavior, exposes
each retrieval and context stage, safely handles insufficient evidence, and can
be evaluated against a frozen benchmark.

### Phase 3 — Hybrid Retrieval

**Status:** Implemented; full frozen-benchmark qualification pending.

**Objective**

Combine semantic and exact-term retrieval while preserving the external Phase 2
pipeline and evaluation contracts.

**Why this phase exists**

Dense retrieval can miss identifiers, acronyms, exact policy language, and rare
domain terms. BM25 complements semantic similarity, while fusion provides a
controlled way to combine differently scored result sets.

**Major capabilities**

- BM25 lexical retrieval;
- parallel dense and BM25 retrieval;
- hybrid retrieval through Reciprocal Rank Fusion;
- tokenizer-aware context budgeting;
- clickable PDF citations;
- a configuration-driven `RunManager`;
- isolated per-run artifact bundles;
- CSV, XLSX, and standalone HTML reports;
- backward-compatible adapters and result schemas; and
- controlled comparison with the frozen Phase 2 dense baseline.

**Expected deliverables**

- A new Phase 3 notebook without modifying frozen notebooks.
- Reusable lexical, hybrid-fusion, token-budget, citation-export, and run
  management modules.
- Reproducible run directories below
  `outputs/batch_answers/03_Hybrid_Retrieval/`.
- Serialized configuration, retrieval traces, metrics, logs, figures, and
  retained context.
- Benchmark reports comparing dense-only and hybrid modes.

**Exit criteria**

- Hybrid retrieval demonstrates a documented improvement or trade-off against
  Phase 2 on the same frozen benchmark.
- Dense-only compatibility remains available.
- Run artifacts are complete, deterministic where practical, and independently
  inspectable.
- Context limits are enforced with the centralized configured tiktoken manager
  rather than a character estimate.
- Existing Phase 1 and Phase 2 tests and notebook contracts remain valid.

Implementation status: the notebook, reusable modules, compatibility paths,
structured logs, reports, artifact bundle, and deterministic tests are present.
The first, empirical exit criterion remains open until the approved local corpus
and models complete the frozen 200-question dense-versus-hybrid comparison.

### Phase 4 — Reranking and Precision Optimization

**Status:** Planned.

**Objective**

Improve precision after hybrid candidate retrieval and reduce irrelevant
evidence entering the context window.

**Why this phase exists**

Hybrid retrieval is designed to improve candidate recall, but fused candidates
may still contain weak or redundant evidence. A local reranker can reorder a
larger candidate pool before context construction.

**Major capabilities**

- a local cross-encoder or equivalent local reranker;
- configurable candidate-pool and final top-k sizes;
- relevance-score and latency inspection;
- removal or demotion of irrelevant chunks; and
- reranked versus non-reranked benchmark comparison.

**Expected deliverables**

- A new Phase 4 notebook and reusable reranking interface.
- Local model configuration and hardware-aware batch settings.
- Precision, citation, answer-quality, latency, and resource comparisons.
- Reports that isolate the effect of reranking from hybrid retrieval.

**Exit criteria**

- The reranker improves defined precision or downstream answer metrics at an
  acceptable measured latency and resource cost.
- Reranking can be disabled through configuration.
- Phase 3 remains a reproducible comparison baseline after its benchmark
  qualification is recorded and the phase is frozen.

### Phase 5 — Agentic RAG and Multi-step Retrieval

**Status:** Planned.

**Objective**

Add bounded, inspectable planning and verification only where deterministic
single-step retrieval is insufficient.

**Why this phase exists**

Complex, multi-part, cross-document, and multi-hop questions may require query
decomposition and repeated evidence gathering. Agent-like components should be
introduced only after retrieval and reranking are strong baselines.

**Major capabilities**

- agentic retrieval planning;
- query decomposition;
- multi-hop and iterative retrieval;
- tool-like retrieval workflows;
- evidence verification;
- critic or verifier roles;
- bounded iteration and explicit stop conditions; and
- strict grounding and safe failure.

**Expected deliverables**

- A new Phase 5 notebook and reusable planner/verifier interfaces.
- Full traces of plans, subqueries, retrieved evidence, decisions, and final
  citations.
- Configuration limits for steps, latency, context, and model calls.
- Comparisons against the frozen non-agentic baseline.

**Exit criteria**

- Multi-step workflows improve selected complex-query metrics without reducing
  grounding or citation quality.
- Every action and evidence transition is auditable.
- Deterministic retrieval remains the default for questions that do not require
  planning.
- Loops, tool access, and generation are bounded by configuration.

### Phase 6 — Production Hardening and Enterprise UX

**Status:** Planned.

**Objective**

Turn validated research modules into a reliable, observable, deployable
on-premises platform foundation.

**Why this phase exists**

Notebook success does not by itself provide operational reliability,
multi-user safety, maintainability, or an enterprise interface.

**Major capabilities**

- performance and memory optimization;
- incremental and resumable indexing;
- stronger artifact and lifecycle management;
- packaged reusable modules and stable service boundaries;
- deployment preparation for organization-controlled infrastructure;
- local observability, health, and audit logging;
- enterprise search and evidence-review UX; and
- access-control and document-isolation planning.

**Expected deliverables**

- Versioned service and package interfaces around validated modules.
- Deployment, configuration, backup, recovery, and operational documentation.
- Performance profiles and capacity guidance.
- Enterprise UX for search, evidence inspection, citations, reports, and human
  review.
- A reviewed authorization model before sensitive multi-user use.

**Exit criteria**

- The platform is reproducibly deployable in a representative offline
  environment.
- Ingestion, indexing, retrieval, and generation have observable failure and
  recovery behavior.
- Security boundaries and access-control requirements are documented and
  tested before production data access.
- Notebook baselines remain available for regression analysis.

### Phase 7 — Enterprise Knowledge OS

**Status:** Long-term target.

**Objective**

Deliver a secure, extensible, on-premises Knowledge OS for trusted interaction
with approved enterprise documentation.

**Why this phase exists**

The final value is not a single RAG pipeline but a governed knowledge platform
that makes answers explainable, reviewable, and operationally dependable.

**Major capabilities**

- on-premises deployment readiness;
- secure enterprise document search;
- explainable, evidence-grounded answers;
- auditable citations and retrieval histories;
- human-review and escalation workflows;
- extensible retrieval and model architecture;
- governed multi-user operation; and
- continuous offline evaluation and regression monitoring.

**Expected deliverables**

- An operational Knowledge OS interface and supporting on-premises services.
- Security, governance, audit, lifecycle, and support procedures.
- Validated ingestion-to-answer traceability.
- Documented extension points for models, retrieval methods, evaluators, and
  enterprise integrations.

**Exit criteria**

- Every answer can be traced to authorized retrieved evidence.
- Unsupported questions fail safely.
- Access, review, audit, and operational controls meet approved enterprise
  requirements.
- New releases are benchmarked against frozen baselines and pass documented
  reliability gates.

## Target Long-Term Architecture

The diagram shows the intended end-state flow. Only loading through dense
retrieval, Phase 2 context construction, local Ollama generation, citations, and
current evaluation/reporting are implemented today. Lexical retrieval,
reranking, agent planning, production interfaces, and several enterprise
controls remain planned. Hybrid lexical/vector retrieval and tokenizer-aware
context construction are implemented; reranking and agent planning are not.

```text
Enterprise Documents
        ↓
Ingestion
        ↓
Chunking
        ↓
Embeddings
        ↓
Dense Index + Lexical Index
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Context Builder
        ↓
Agent Planner / Verifier
        ↓
Grounded Local LLM
        ↓
Citation Engine
        ↓
Reports / Dashboard / Knowledge OS Interface
```

Each stage should expose a stable contract, accept configuration, retain
traceable metadata, and remain replaceable without forcing unrelated stages to
change.

## Deferred Ideas and Future Research

These topics are intentionally deferred. They require separate evidence,
prioritization, and design review before entering an active phase:

- ColBERT or other late-interaction retrieval;
- knowledge graph retrieval;
- multi-vector indexing;
- document versioning and supersession rules;
- retrieval-time access control;
- incremental and event-driven indexing;
- a local HTTP server for clickable PDF citations;
- Model Context Protocol (MCP) integration;
- user feedback loops;
- active learning from failed or weak queries;
- advanced semantic, retrieval, faithfulness, and citation metrics; and
- secure multi-user deployment.

Listing an idea here does not authorize implementation or make it part of the
current architecture.

## Phase Governance Rules

- Do not modify a previous notebook after its phase is frozen.
- Use a new notebook and reusable modules for each new phase.
- Extend existing modules through configuration or compatible adapters rather
  than breaking public contracts.
- Benchmark every phase against the previous frozen baseline using the same
  applicable dataset and clearly documented configuration.
- Produce reproducible, non-overwriting artifacts for every benchmarked phase.
- Record effective configuration, retrieval traces, citations, metrics, and
  relevant environment details.
- Keep planned capabilities labeled as planned until implementation and
  verification are complete.
- Update requirements, current-state documentation, roadmap status, and
  operational instructions before major implementation begins and again when a
  phase is frozen.

## Success Definition

This roadmap succeeds when CIAL Knowledge OS evolves into a reliable, fully
offline, auditable, enterprise-ready platform where every answer is grounded in
retrieved evidence, every citation is traceable, unsupported questions fail
safely, and every architectural improvement is measured against a frozen
baseline.
