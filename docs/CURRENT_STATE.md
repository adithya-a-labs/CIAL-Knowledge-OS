# CIAL Knowledge OS: Current State through Phase 5

Last audited: 2026-07-06

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
Enterprise documents in configured data/files/ taxonomy
  -> Enterprise File Format Registry validation and readiness scan
  -> recursive supported-document loading plus OCR-supported image extraction
  -> metadata-preserving chunking
  -> local embeddings
  -> embedded local Qdrant (default) or local Docker Qdrant server (opt-in)
  -> selectable dense, BM25, or hybrid retrieval
  -> Reciprocal Rank Fusion in hybrid mode
  -> Phase 2 query variants and multi-query evidence collection
  -> Phase 4 local cross-encoder reranking
  -> explainable evidence selection
  -> overlap merging and token-aware or character-compatible context construction
  -> grounded local Ollama generation
  -> clickable citations, evidence quality, traces, run bundles, and offline evaluation
```

Notebooks are the learning and orchestration layer. Reusable behavior belongs in
`src/cial_knowledge_os/`, where ingestion, chunking, embeddings, vector storage,
retrieval, generation, context construction, evaluation, exports, and
visualization are split into focused modules. Configuration is centralized in
`KnowledgeOSConfig`, `Phase2Config`, `Phase3Config`, and `Phase4Config`;
experiment sweeps add declarative `ExperimentConfig` and `ExperimentGrid`
values.

`KnowledgeOSConfig.knowledge_root` resolves to `data/files/`, the canonical
enterprise source repository. Files may be nested to any depth; the first folder
is recorded as `category` and the second, when present, as `collection`.
Recommended top-level categories include `aviation`, `cybersecurity`,
`engineering`, `hr`, and `legal`, with standards bodies, systems, or document
sets as second-level collections. Discovery is governed by the Enterprise File
Format Registry, which classifies files as `SUPPORTED_NOW`, `OCR_SUPPORTED`,
`RECOGNIZED_FUTURE_SUPPORT`, or `UNSUPPORTED`. Only `SUPPORTED_NOW` and
`OCR_SUPPORTED` files enter extraction, chunking, embedding, and indexing.
Recognized future and unsupported files are logged, skipped, and reported.

Currently processed formats include PDF, DOCX, DOC, XLSX, XLS, CSV, PPTX, PPT,
TXT, Markdown, HTML, JSON, XML, and YAML. PNG, JPG/JPEG, and TIFF are processed
through the modular OCR subsystem before chunking. Recognized future categories
include email, archives, source code, configuration and DevOps files,
audio/video, and CAD/engineering formats.

Runtime ingestion does not search the former `data/pdf/` corpus. Missing or
empty canonical repositories produce an empty corpus rather than a fallback.
`scripts/migrate_pdf_to_files.py` remains a one-time transition utility: it
copies legacy PDFs to `data/files/legacy_pdf/` by default and supports
`--dry-run` and explicit `--move` modes.

Indexing scans the canonical root on startup and compares SHA-256 fingerprints
with `data/indexes/document_manifest.json`. Only new and changed processable
registry files are loaded, chunked, and embedded. Changed and deleted document
points are removed from Qdrant before upsert. The complete stored chunk corpus
is then reused for neighbor operations and BM25, which is rebuilt safely when
corpus membership changes. Indexing, file-format readiness, and OCR summaries
are logged and added to Phase 4 summary/metrics/HTML/CSV/XLSX artifacts.
`incremental_indexing_enabled=False` preserves full processing;
`force_rebuild_index=True` recreates the configured collection. The manifest is
collection-bound, so switching collection names safely causes a full scan.

The current LLM adapter uses Ollama. The surrounding pipeline accepts replaceable
local model objects, but adapters for other local runtimes such as vLLM and
llama.cpp are still future work.

### Qdrant deployment modes

`KnowledgeOSConfig` provides `qdrant_mode`, `qdrant_url`,
`qdrant_api_key`, `qdrant_collection_name`, `qdrant_dir`,
`qdrant_batch_size`, and `qdrant_upsert_wait`. Omitting `qdrant_mode`
preserves the `embedded` default and the existing path-based client.
`qdrant_mode="server"` uses the configured URL and performs a health probe
before indexing or retrieval. An unreachable service reports local Docker
start commands rather than a low-level connection error.

Indexing constructs and submits bounded point batches rather than serializing
the full corpus update in one JSON request. The resolved default is 256 points
per embedded upsert and 32 points per server upsert. Server deployments should
retain a small batch unless measured vector and payload sizes justify a larger
value. `qdrant_upsert_wait=True` preserves synchronous completion semantics and
can be disabled explicitly when the caller accepts asynchronous server
processing. Every completed batch emits structured progress without changing
point IDs, vectors, or payload metadata.

Embedded mode remains appropriate for notebooks, demonstrations, and small
collections. The local Docker server is recommended for large collections and
concurrent clients:

```bash
docker compose -f docker-compose.qdrant.yml up -d
curl http://localhost:6333/healthz
```

Set `qdrant_mode="server"` and
`qdrant_url="http://localhost:6333"` in the active phase configuration. Stop
and start it with Docker Compose. Storage remains on-premises in the Docker
named volume `cial_qdrant_storage`; there is no Qdrant Cloud dependency. The
named volume replaces the previous Windows bind mount that could fail Qdrant
segment renames during optimization.

Read-only health inspection reports reachability, collection presence, point
and indexed-vector counts, collection and optimizer status, and embedding
dimension compatibility. Infrastructure preflight also reports offline model
flags, RAM, disk, configured GPU availability, and local Ollama availability.
A red optimizer is a warning rather than an automatic crash; an unreachable
opt-in server is a preflight error.

Existing embedded collections can be inspected or migrated without changing
the frozen notebooks:

```bash
python scripts/migrate_embedded_qdrant_to_server.py \
  --source data/qdrant/cial_phase4 \
  --url http://localhost:6333 \
  --collection cial_phase4 \
  --batch-size 512
```

The utility preserves point IDs, dense/named/sparse vector configuration,
vectors, and payloads, supports `--dry-run`, and requires `--force` before
overwriting a target collection.

## Completed Phase 1: Basic RAG

The frozen Phase 1 baseline is represented by
`notebooks/01_Basic_RAG.ipynb` and `BasicRAGPipeline`. It implements:

- recursive, configuration-driven local PDF loading with Docling and a PyMuPDF
  fallback;
- local text loading and metadata-preserving chunking;
- local SentenceTransformers embeddings;
- persistent embedded Qdrant storage;
- manifest-driven incremental Qdrant updates and safe full rebuilds;
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
Phase 2 still enforces its frozen character budget. Its backward-compatible
export field now receives an exact centralized tiktoken count rather than the
former tokenizer-independent estimate.

## Implemented Phase 3: Hybrid Retrieval

`notebooks/03_Hybrid_Retrieval.ipynb` and `Phase3RAGPipeline` now implement:

- a common retriever protocol with dense and `rank-bm25` implementations;
- dense-only, BM25-only, and hybrid modes;
- proper configurable Reciprocal Rank Fusion with modality ranks and scores;
- shared chunk reuse plus fingerprinted BM25 token caching;
- centralized tiktoken-aware context fitting, truncation, and reporting;
- a packaged hash-verified `cl100k_base` vocabulary for offline tokenization;
- the unchanged Phase 2 character budget when token budgeting is disabled;
- metadata-derived `file://` and configurable localhost PDF page links;
- configurable structured logging;
- a `RunManager` with collision-safe timestamped directories;
- backward-compatible CSV plus formatted XLSX and standalone HTML reports;
- configuration, summary, retrieval, metrics, logs, figures, and per-question
  context artifacts; and
- per-question execution traces from query variants through dense/BM25
  candidates, RRF, post-processing, token budgeting, generation, citations, and
  artifact export, with notebook and standalone HTML diagnostics; and
- additive integration with the existing batch and evaluation contracts.

The implementation and offline tests are complete. Phase 3 is not yet a frozen
baseline because the full 200-question local-model comparison has not been run
in this checkout. The empirical exit gate remains a documented Phase 2 versus
Phase 3 improvement or trade-off on the unchanged frozen benchmark.

## Implemented Phase 4: Reranking and Evidence Selection

`notebooks/04_Reranking_and_Evidence_Selection.ipynb` and
`Phase4RAGPipeline` now implement:

- a lazy SentenceTransformers cross-encoder with configurable model, CPU/GPU
  device, batch size, and cache/download policy;
- cache-first loading on every process, automatic one-time staging on a
  developer cache miss, and strict no-network enterprise mode through
  `reranker_local_files_only=True`;
- a deterministic `MockReranker` with the same interface for unit and
  end-to-end tests;
- reranking after RRF, without directly averaging dense, BM25, RRF, and
  cross-encoder scores from incompatible scales;
- evidence selection using maximum count, reranker threshold, source-diversity
  cap, lexical redundancy reduction, and evidence-token budget strategies;
- a default three-chunk evidence floor, eight-chunk ceiling, adaptive ranked
  fallback, and an 800--1500 selected-token target for normal QA;
- explicit weak/mixed/strong evidence confidence plus an evidence-sufficiency
  gate that blocks extractive fallback for weak, all-fallback, or
  below-minimum-score evidence;
- deterministic current/live/external-data detection, with an
  `unsupported_query` result when indexed evidence does not directly support
  the request;
- a Phase 4-specific detailed answer style that retains Phase 3 grounding and
  citations while selecting question-relevant section families for findings,
  implications, controls, procedures, comparisons, risks/gaps, caveats, and
  next actions from selected evidence only;
- configurable `answer_detail_level`, `min_answer_words`, optional
  `max_answer_words`, `prefer_structured_answers`,
  `adaptive_answer_sections`, and `include_decision_notes` generation policy;
- Phase 4-only generation retries and cooldown for retryable local Ollama
  runner, stream, transport, HTTP 500, and memory-allocation failures;
- per-question CSV/JSONL checkpoints and indexed question-hash resume support,
  including duplicate-text safety and final failure-row preservation;
- an intentional smaller evidence budget before the existing final context
  budget;
- per-chunk evidence strength, retrieval provenance, citation availability,
  metadata completeness, and source-diversity diagnostics;
- candidate, selected, and final-context tokens; token-reduction percentage;
  discard counts and reasons; and stage latency;
- full and compact serializable traces showing retrieved, fused, reranked,
  selected, context, answer, citations, and artifact paths;
- `smoke`, `manual_qa`, `benchmark`, and `export_only` runner modes with a
  notebook-safe manual-question limit;
- additive Phase 4 CSV fields, manager-friendly XLSX, standalone offline HTML,
  JSON, log, context, and SVG decision artifacts;
- safe inline HTML citation badges for numeric and source/page/chunk answer
  markers, with compact fallback chips and collapsible citation details while
  preserving CSV/XLSX citation columns; and
- dependency-injected pipeline/reporting interfaces suitable for future
  automated benchmark execution without notebook execution.

Phase 4 reuses the Phase 3 retrievers, RRF, token manager, context builder,
citation engine, `RunManager`, batch collector, and evaluation interfaces.
Earlier classes and configuration defaults are unchanged. Phase 4 disables
neighbor expansion by default so evidence that did not pass reranking is not
introduced after selection; callers can opt in explicitly.

Long terminal runs persist `partial_results.csv`, `partial_results.jsonl`,
`partial_retrieval.jsonl`, and `checkpoint.json` after every attempt. Resume
uses the same run directory and question order, skips successful occurrences,
and retries failed or interrupted occurrences. Standard final artifacts are
regenerated even when some generation attempts remain failed.

Phase 4 distinguishes four answer outcomes. `answered` means generation
succeeded or a generator refusal was replaced by a sufficient-evidence
extractive fallback. `insufficient_evidence` means the indexed evidence cannot
support an answer or safe fallback. `unsupported_query` means the question
appears to require current/live/external data absent from the indexed evidence.
`generation_failed` means the local generator exhausted retries before a
response artifact could be completed.

Phase 4 evidence reduction is not an answer-length optimization. Detailed
synthesis is generated from the smaller, higher-precision selected context.
Word targets apply only when evidence supports them and do not permit padding
or unsupported claims.

Phase 4 now supports semi-adaptive answer sections. The default prompt chooses
only section families that fit the question shape; setting
`adaptive_answer_sections=False` restores the previous fixed template for
reproducibility. This Phase 4 mechanism is not the full response planner; the
optional Phase 5 layer now provides that planner. Neither change establishes a
benchmark-qualified quality improvement.

The default `reranker_local_files_only=False` is a developer-experience policy,
not a cloud-inference dependency. Loading remains lazy and always tries the
local Hugging Face cache first. Only a cache miss permits a model download,
which is cached for subsequent offline runs. Enterprise deployments set the
field to `True`; a missing model then produces an actionable error without any
network attempt. Automated tests continue to inject `MockReranker`.

The implementation and deterministic automated tests are complete. Phase 4 is
**implemented but not benchmark-qualified** because the full unchanged
200-question Phase 3 Hybrid versus Phase 4 Reranked Hybrid comparison has not
been run with the approved local corpus and models. No benchmark improvement is
claimed in this state document.

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
Corrections or extensions require a new version. Phase 3 and Phase 4
comparisons must retain the existing version so retrieval, answer, citation,
unsupported-question, token, and latency results remain comparable.

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

Some directories are created on demand and may be empty in a checkout. Later
phases extend this `outputs/` hierarchy and do not introduce a competing
top-level `artifacts/` directory.

## Phase Isolation and Frozen Notebook Policy

- Do not modify completed Phase 1 or Phase 2 notebooks.
- `01_Basic_RAG.ipynb` and
  `02_Query_Transformations_and_Context_Construction.ipynb` are frozen,
  runnable baselines.
- The Phase 2 automated-evaluation notebook is also a completed orchestration
  baseline and should remain reproducible.
- Add each new capability through a new phase notebook and reusable source
  modules.
- `notebooks/04_Reranking_and_Evidence_Selection.ipynb` is the Phase 4
  engineering and qualification notebook. It does not modify earlier notebooks.
- Preserve existing notebook imports, configuration defaults, output schemas,
  and runnable behavior unless a documented compatibility migration is
  unavoidable.
- Placeholder notebooks for later phases do not imply that those phases are
  implemented.

## Current Limitations

The current implementation has:

- no full-corpus Phase 3 or Phase 4 benchmark qualification;
- no calibrated semantic relevance/entailment evaluator;
- no retrieval-time authorization enforcement;
- no visual document understanding;
- no multimodal retrieval;
- no contradiction detection;
- no calibrated Phase 5 readiness or consensus thresholds; and
- no completed real-model Phase 5 benchmark qualification.

The repository does generate a self-contained HTML evaluation dashboard. That is
an aggregate evaluation report and remains separate from the implemented
standalone per-run Phase 3 and Phase 4 reports.

## Phase 3 and Phase 4 Qualification Status

Implemented capabilities are covered by deterministic offline tests. Remaining
qualification work is to run dense and hybrid modes against the same frozen
benchmark with the approved local corpus/models, retain both artifact sets, and
document quality, safety, token, and latency trade-offs.

Phase 4 unit, integration, serialization, compatibility, and artifact tests are
implemented. Remaining qualification work is to run the unchanged frozen
benchmark in Phase 3 Hybrid and Phase 4 Reranked Hybrid modes, retain both run
bundles, and compare answer quality, citation quality, unsupported-question
behavior, context tokens, token reduction, latency, selected/discarded chunks,
average reranker score, and evidence-strength distribution.

Visual document understanding, multimodal retrieval, and contradiction
detection are reserved for Phase 4.5. They are deferred, not partially
implemented by Phase 4.

The implemented Phase 3 output contract is:

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
            |-- partial_results.csv
            |-- partial_results.jsonl
            |-- partial_retrieval.jsonl
            |-- checkpoint.json
            |-- figures/
            `-- context/
```

Exact naming and collision behavior must be defined by configuration and the
`RunManager`, not repeated as ad hoc notebook logic.

The additive Phase 4 output contract is:

```text
outputs/
`-- batch_answers/
    `-- 04_Reranking_and_Evidence_Selection/
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

`retrieval.json` is the source of truth for per-question candidate ranks,
reranker scores, selected/discarded decisions and reasons, evidence quality,
token use, latency, citations, answer, and artifact paths. CSV contains compact
machine-readable summaries. XLSX retains clickable evidence links. HTML embeds
decision visualizations and requires no CDN, external JavaScript, or network
access.

## Backward Compatibility Policy

Later phases may introduce a more efficient internal architecture, but callers,
notebooks, exports, and evaluation tooling must retain their existing contracts:

> New architecture internally. Same contracts externally.

Additive fields and adapters are preferred. If a contract must change, document
the migration, retain a compatibility path, and compare behavior against the
frozen Phase 2 baseline.

## Configuration Policy

Operational choices must not be scattered as literals through notebooks or
pipeline logic. In particular, Phase 3 and Phase 4 must not hardcode:

- paths;
- Qdrant deployment mode, URL, API key, collection, or embedded directory;
- model names;
- output folders;
- retrieval modes;
- token budgets; or
- artifact filenames.

Phase 4 additionally configures the reranker model/device/batch/local-only
policy, candidate depth, selection strategies, minimum/maximum evidence count,
reranker threshold, adaptive fallback, weak-evidence answer policy, source cap,
redundancy threshold, evidence token budget and target range, evidence strength
thresholds, minimum extractive-fallback score, weak-fallback override,
unsupported-query detection, run mode, trace mode, and large-run guard.
Phase 4 generation reliability also configures retry count, retry cooldown, and
an optional answer-word upper bound. These controls apply only to Phase 4;
earlier phase defaults and generation behavior remain unchanged.

Discard reasons are normalized as `threshold_failed`, `redundancy`,
`source_diversity_limit`, `token_budget`, `empty_text`, and
`lower_rank_fallback`. Aggregate counts are persisted in `metrics.json` and
rendered in the standalone report. Diagnostics flag candidate starvation,
greater-than-90-percent reduction, under-500-token answered contexts, and zero
average selected score with non-empty candidates.

Expose these through typed configuration or explicit function arguments, validate
them at the boundary, serialize the effective configuration with every run, and
use one resolved configuration throughout the run. Central default values are
acceptable; hidden duplicated constants are not.
# Phase 5 additive implementation

Adaptive Agentic Response Planning is implemented as an optional layer over
Phase 4. It is disabled by default, uses injected local model clients, accepts
text-only Phase 4 evidence without migration, and can preserve optional visual,
table, diagram, screenshot, scanned-region, caption, and OCR fields. Its HTML
report adds computed decision-readiness diagnostics without changing previous
reports or notebooks.

## Cross-phase execution observability

The Execution & Observability Framework is implemented under
`src/cial_knowledge_os/execution/`. Phase 4 batch execution emits run, question,
retrieval, reranking, evidence-selection, generation, checkpoint, and export
events. Shared indexing emits start/progress/completion events, preflight emits
health events, and Phase 5 mirrors its existing agent and consensus lifecycle
into EOF.

EOF writes `execution_trace.jsonl`, `progress.json`, and `progress.log` under
`outputs/runs/<run_id>/`. Rich, psutil, and nvidia-smi are optional and degrade
to plain or omitted output. The framework has no cloud dependency and no
authority over pipeline control flow. Existing artifacts and checkpoint/resume
behavior remain the source of truth for work products.
