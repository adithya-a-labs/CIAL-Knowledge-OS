# CIAL Knowledge OS: Current State and Phase 3 Baseline

Last audited: 2026-07-03

This document describes the implemented repository state. `PROJECT_REQUIREMENTS.md`
defines the binding requirements, while this file distinguishes completed
capabilities from planned work.

## Project Overview

CIAL Knowledge OS is an enterprise-grade, fully offline, notebook-first
retrieval-augmented generation (RAG) platform for enterprise documentation. The
current repository is an experimentation and reusable-module foundation, not yet
a production service or user interface.

The platform is designed around these principles:

- offline-first operation on organization-controlled infrastructure;
- open-source and open-weight local models;
- no cloud inference or cloud vector database;
- notebook-first, inspectable experimentation;
- reusable production-oriented modules under `src/cial_knowledge_os/`;
- configuration-driven and reproducible behavior;
- model-agnostic interfaces;
- token-efficient evidence selection and prompting; and
- enterprise-ready traceability, safe failure, and extension boundaries.

## Current Architecture

```text
Local documents
  -> PDF/text loading
  -> metadata-preserving chunking
  -> local embeddings
  -> embedded local Qdrant
  -> dense retrieval
  -> Phase 2 query variants and multi-query evidence collection
  -> deduplication and optional neighbor expansion
  -> overlap merging and character-bounded context construction
  -> grounded local Ollama generation
  -> citations, traces, batch exports, and offline evaluation
```

Notebooks are the learning and orchestration layer. Reusable behavior belongs in
`src/cial_knowledge_os/`, where ingestion, chunking, embeddings, vector storage,
retrieval, generation, context construction, evaluation, exports, and
visualization are split into focused modules. Configuration is centralized in
`KnowledgeOSConfig` and `Phase2Config`; experiment sweeps add declarative
`ExperimentConfig` and `ExperimentGrid` values.

The current LLM adapter uses Ollama. The surrounding pipeline accepts replaceable
local model objects, but adapters for other local runtimes such as vLLM and
llama.cpp are still future work.

## Completed Phase 1: Basic RAG

The frozen Phase 1 baseline is represented by
`notebooks/01_Basic_RAG.ipynb` and `BasicRAGPipeline`. It implements:

- local PDF loading with Docling and a PyMuPDF fallback;
- local text loading and metadata-preserving chunking;
- local SentenceTransformers embeddings;
- persistent embedded Qdrant storage;
- dense semantic retrieval;
- local Ollama generation;
- bounded grounded prompts and safe prompt instructions;
- source, page, chunk, score, and metadata-aware citations;
- basic latency benchmarking and visualizations;
- versioned batch answer CSV exports; and
- a modular package under `src/cial_knowledge_os/`.

Phase 1 is a dense top-k baseline. It does not implement the production features
listed under **Current Limitations**.

## Completed Phase 2: Query Transformations and Context Construction

The frozen Phase 2 baseline is represented by
`notebooks/02_Query_Transformations_and_Context_Construction.ipynb`,
`notebooks/testing/Phase2_Automated_Evaluation.ipynb`, and
`Phase2RAGPipeline`. It implements:

- inspectable original, rewritten, keyword-expanded, and domain-reformulated
  query variants;
- configurable multi-query dense retrieval;
- evidence fusion by collecting results from all enabled variants;
- exact deduplication by `(source, page, chunk_id)`, retaining the strongest
  score and query provenance;
- source-relative neighbor expansion;
- overlap-aware merging of contiguous chunks;
- character-bounded context construction and compression;
- one stronger grounded generation pass over the constructed context;
- explicit insufficient-evidence handling with no citations on safe failure;
- citation mapping to the final compressed evidence;
- retrieval, source, score, context, citation, answer-status, and latency
  visualizations;
- versioned batch exports with Phase 2 trace columns;
- a deterministic automated evaluation framework;
- a frozen 200-question CISG benchmark; and
- unit and regression tests for pipeline, export, visualization, and evaluation
  behavior.

The current query rewrite is deterministic string normalization. It does not call
an LLM. `QueryTransformer` supports registered local strategies, so an AI-based
rewrite can be introduced later without changing its external role. Likewise,
Phase 2 estimates tokens for reporting but enforces a character budget rather
than a tokenizer-aware token budget.

## Evaluation Framework

The reusable evaluation framework is under `src/cial_knowledge_os/`:

| Module | Purpose |
|---|---|
| `benchmark_loader.py` | Loads benchmark CSV rows and optional metadata into typed benchmark records. |
| `evaluation_metrics.py` | Applies deterministic keyword, safe-failure, hallucination, and citation heuristics; aggregates and ranks experiments. |
| `evaluation_report.py` | Builds recommendations and writes the Markdown recommendation report. |
| `experiment_config.py` | Defines immutable experiment configurations, Cartesian grids, and stable configuration fingerprints. |
| `experiment_runner.py` | Runs every benchmark question for each configuration, isolates question failures, writes experiment and summary CSVs, and coordinates reports. |
| `visualization_dashboard.py` | Reads evaluation artifacts and generates a self-contained offline HTML dashboard with embedded data and no external scripts. |

`visualization.py` separately provides pandas and matplotlib diagnostics for
interactive notebook analysis. `batch_qa.py` provides general versioned batch
answer exports; it is not a substitute for ground-truth evaluation.

Evaluation is deterministic and offline, but currently heuristic. It does not
provide semantic entailment, retrieval recall against labeled relevant chunks,
or model-judged correctness.

## Benchmark Structure and Policy

The current benchmark is:

```text
data/benchmarks/cisg/
|-- benchmark_answers.csv
|-- benchmark_metadata.json
|-- cisg_questions_v1.txt
|-- README.md
`-- CHANGELOG.md
```

`benchmark_answers.csv` contains 200 questions spanning factual, definition,
procedure, comparison, executive-summary, enterprise, cross-document, and
unsupported categories. Metadata identifies it as `cisg_benchmark_v1`, version
`1.0.0`, with status `frozen`.

The benchmark dataset is immutable. Do not edit it to accommodate a new phase.
Corrections or extensions require a new version, while Phase 3 comparisons must
retain the existing version for a fair comparison with the frozen Phase 2
baseline.

## Current Output Structure

The standard repository-local output roots are:

```text
outputs/
|-- batch_answers/
|-- benchmarks/
|-- evaluations/
|-- exports/
`-- logs/
```

Current batch and Phase 2 evaluation artifacts live below
`outputs/batch_answers/`. An evaluation sweep creates:

```text
outputs/batch_answers/<phase>/
|-- experiments/
|   `-- experiment_001.csv
|-- summary/
|   `-- experiment_summary.csv
`-- reports/
    |-- recommendation.md
    `-- dashboard.html
```

Some directories are created on demand and may be empty in a checkout. Phase 3
must extend this `outputs/` hierarchy and must not introduce a new top-level
`artifacts/` directory.

## Phase Isolation and Frozen Notebook Policy

- Do not modify completed Phase 1 or Phase 2 notebooks.
- `01_Basic_RAG.ipynb` and
  `02_Query_Transformations_and_Context_Construction.ipynb` are frozen,
  runnable baselines.
- The Phase 2 automated-evaluation notebook is also a completed orchestration
  baseline and should remain reproducible.
- Add each new capability through a new phase notebook and reusable source
  modules.
- Preserve existing notebook imports, configuration defaults, output schemas,
  and runnable behavior unless a documented compatibility migration is
  unavoidable.
- Placeholder notebooks for later phases do not imply that those phases are
  implemented.

## Current Limitations

The current implementation has:

- dense retrieval only;
- no BM25 or other lexical retrieval;
- no hybrid lexical/vector retrieval;
- no Reciprocal Rank Fusion (RRF);
- no reranking stage;
- no tokenizer-aware context budgeting;
- no clickable citation export contract;
- no per-run XLSX export;
- no complete standalone per-run HTML artifact bundle;
- no `RunManager` abstraction; and
- no completed Phase 3 implementation.

The repository does generate a self-contained HTML evaluation dashboard. That is
an aggregate evaluation report, not the planned per-run Phase 3 artifact bundle.
Current CSV citations are structured JSON values but are not exported as
clickable links.

## Phase 3 Roadmap

Phase 3 should add and evaluate:

- BM25 lexical retrieval;
- hybrid dense and lexical retrieval;
- Reciprocal Rank Fusion;
- tokenizer-aware context budgeting;
- clickable citations;
- a `RunManager` for isolated, reproducible run directories;
- per-run CSV, XLSX, and standalone HTML reports;
- machine-readable configuration, summary, retrieval, and metric artifacts;
- logs, figures, and retained context evidence; and
- controlled comparisons against the frozen Phase 2 dense baseline.

Reranking remains absent from the current system. Unless Phase 3 scope is
explicitly expanded, keep it as the next post-hybrid phase rather than silently
mixing it into the Phase 3 comparison.

The intended Phase 3 output contract is:

```text
outputs/
`-- batch_answers/
    `-- 03_Hybrid_Retrieval/
        `-- run_<timestamp>/
            |-- results.csv
            |-- results.xlsx
            |-- report.html
            |-- config.json
            |-- summary.json
            |-- retrieval.json
            |-- metrics.json
            |-- logs.txt
            |-- figures/
            `-- context/
```

Exact naming and collision behavior must be defined by configuration and the
`RunManager`, not repeated as ad hoc notebook logic.

## Backward Compatibility Policy

Phase 3 may introduce a more efficient internal architecture, but callers,
notebooks, exports, and evaluation tooling must retain their existing contracts:

> New architecture internally. Same contracts externally.

Additive fields and adapters are preferred. If a contract must change, document
the migration, retain a compatibility path, and compare behavior against the
frozen Phase 2 baseline.

## Configuration Policy

Operational choices must not be scattered as literals through notebooks or
pipeline logic. In particular, Phase 3 must not hardcode:

- paths;
- model names;
- output folders;
- retrieval modes;
- token budgets; or
- artifact filenames.

Expose these through typed configuration or explicit function arguments, validate
them at the boundary, serialize the effective configuration with every run, and
use one resolved configuration throughout the run. Central default values are
acceptable; hidden duplicated constants are not.

