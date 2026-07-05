
# CIAL Knowledge OS

An enterprise-grade, fully offline, notebook-first RAG platform for enterprise
documentation. The current repository provides completed dense-retrieval Phase
1 and query/context-construction Phase 2 baselines plus implemented Phase 3
hybrid retrieval, Phase 4 reranking/evidence selection, and optional Phase 5
adaptive response-planning architectures. Phases 3–5 still await full local
benchmark qualification. Access control, multimodal retrieval, contradiction
detection, and production applications remain target capabilities.

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

The Phase 3 pipeline adds local BM25, Reciprocal Rank Fusion, tokenizer-aware
context limits, clickable PDF citations, structured logging, and isolated
CSV/XLSX/HTML/JSON run bundles. Phase 1 and Phase 2 remain unchanged baselines.

Phase 4 adds a configurable local cross-encoder after RRF, deterministic mock
reranking for tests, explainable evidence selection, evidence-quality scoring,
candidate-to-context token reduction, and richer standalone run diagnostics.
Implementation and automated-test readiness are complete; full benchmark
qualification is pending.

Corpus indexing is incremental by default across all phases. A SHA-256 document
manifest at `data/indexes/document_manifest.json` tracks canonical
`data/files/` documents and chunk counts. Unchanged PDFs reuse their Qdrant
vectors; changed or deleted document points are removed by stable document
identity, and BM25 is rebuilt safely from the complete stored corpus after a
change. Set `force_rebuild_index=True` (or `FORCE_REBUILD_INDEX = True` in the
Phase 4 batch script) for a full rebuild. Setting
`incremental_indexing_enabled=False` retains full processing behavior.

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
Phase 3 and Phase 4 are implemented in reusable modules and their phase
notebooks. Their full frozen-benchmark quality gates must be run with the
configured local corpus, embedding model, reranker, and Ollama model before
either is described as benchmark-qualified.

See `docs/CURRENT_STATE.md` for the audited architecture, limitations, output
contracts, frozen notebook policy, and qualification roadmap.

## Local Setup

Python 3.11 or newer is required. Install the pinned local stack and the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

On an isolated host, stage approved wheels first and replace the first command with
`python -m pip install --no-index --find-links <wheelhouse> -r requirements.txt`.

The official hash-verified `cl100k_base` tiktoken vocabulary is packaged with
the Python module, so token counting does not make a network request.

The embedding model must already exist in the local Hugging Face cache, and the
configured Ollama model must already exist in the local Ollama store. The Phase
4 reranker is different by design: developer mode checks the local cache first
and automatically downloads/caches a missing reranker once. Enterprise
deployments set `reranker_local_files_only=True` to prohibit network access and
require the approved cache to be staged in advance. The pipeline uses
`BAAI/bge-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, and `gemma3:12b` by
default. Documents and prompts are never sent to a hosted inference service.

### Local Qdrant backends

Embedded Qdrant remains the default for backward compatibility and is suitable
for notebooks, demos, and small corpora. For large corpora, run Qdrant as a
local Docker service:

```bash
docker compose -f docker-compose.qdrant.yml up -d
curl http://localhost:6333/healthz
```

Select the service explicitly in configuration:

```python
from cial_knowledge_os import Phase4Config

config = Phase4Config(
    qdrant_mode="server",
    qdrant_url="http://localhost:6333",
)
```

For `scripts/run_phase4_batch.py`, the same opt-in is available through the
`QDRANT_MODE`, `QDRANT_URL`, and `QDRANT_API_KEY` values in its user
configuration section.

The defaults are `qdrant_mode="embedded"`,
`qdrant_url="http://localhost:6333"`, `qdrant_api_key=None`, and the existing
per-collection `qdrant_dir`. Phase 4 continues to use collection
`cial_phase4`. Server mode changes only the client connection; collection
creation, incremental indexing, stable point IDs, vectors, and payload metadata
retain the same behavior.

Stop and restart the local service with:

```bash
docker stop cial-qdrant
docker start cial-qdrant
```

Migrate an existing embedded Phase 4 collection after starting the server:

```bash
python scripts/migrate_embedded_qdrant_to_server.py \
  --source data/qdrant/cial_phase4 \
  --url http://localhost:6333 \
  --collection cial_phase4 \
  --batch-size 512
```

Use `--dry-run` to inspect the source without changing the server. Use
`--force` only when the target collection may be deleted and recreated. The
Docker service stores data at `data/qdrant_server/`; it remains fully local,
offline-capable, and on-premises. Qdrant Cloud is not used.

Place approved enterprise documents in the canonical `data/files/` knowledge
repository. Discovery is recursive and configured through
`KnowledgeOSConfig.knowledge_root`; ingestion must not hardcode this path. A
recommended taxonomy uses the first directory as the category and the second as
the collection:

```text
data/files/
|-- aviation/icao/
|-- cybersecurity/nist/
|-- engineering/electrical/
|-- hr/
`-- legal/
```

PDF is the currently implemented enterprise-document type. `.txt`, `.md`,
`.docx`, and `.html` are recognized for future loaders and skipped with a clear
log message until implemented. PDF ingestion prefers Docling and falls back to
PyMuPDF. Loaded metadata includes the source filename, absolute path, path
relative to `knowledge_root`, category, collection, extension, and page number
when available, while retaining legacy Phase 1--4 source fields.

Runtime ingestion never searches `data/pdf/`. If `data/files/` is missing or
contains no implemented documents, the pipeline reports an empty corpus. Before
deleting an old `data/pdf/` directory, copy its PDFs into the canonical
repository with:

```bash
python scripts/migrate_pdf_to_files.py --dry-run
python scripts/migrate_pdf_to_files.py
# Destructive source cleanup is explicit:
python scripts/migrate_pdf_to_files.py --move
```

Migration preserves filenames under `data/files/legacy_pdf/` and skips existing
destinations. The old directory can be deleted after migration. Non-sensitive
text fixtures remain in `data/sample/`, temporary
text input remains in `data/raw/`, and experiment Qdrant data is written beneath
`data/qdrant/`; runtime data must not be committed.

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

Phase 3 composes retrievers behind a small protocol and reuses the Phase 2
ingestion, chunking, query transformation, post-processing, generation, export,
and evaluation contracts:

```python
from cial_knowledge_os import Phase3Config, Phase3RAGPipeline, Phase3Runner

config = Phase3Config(
    retrieval_mode="hybrid",
    dense_top_k=10,
    bm25_top_k=10,
    rrf_k=60,
    max_context_tokens=4096,
)
pipeline = Phase3RAGPipeline(config)
pipeline.load()
pipeline.chunk()
pipeline.embed()
pipeline.index()

result = Phase3Runner(pipeline=pipeline, config=config).run(
    questions=["What exact control applies?"],
)
print(result.paths.report_html)
```

Phase 4 extends that pipeline without changing earlier classes:

```python
from cial_knowledge_os import Phase4Config, Phase4RAGPipeline, Phase4Runner

config = Phase4Config(
    project_root=PROJECT_ROOT,
    reranker_model_name="<reranker-model>",
    reranker_local_files_only=False,  # cache first; download once if missing
    reranker_batch_size=16,
    reranker_candidate_top_k=30,
    min_selected_evidence=3,
    max_selected_evidence=8,
    reranker_score_threshold=-4.0,
    fallback_to_top_n_if_empty=True,
    fallback_top_n=3,
    weak_evidence_answer_allowed=True,
    min_fallback_reranker_score=0.35,
    allow_extractive_fallback_for_weak_evidence=False,
    unsupported_query_detection_enabled=True,
    answer_detail_level="detailed",
    min_answer_words=250,
    max_answer_words=None,
    prefer_structured_answers=True,
    adaptive_answer_sections=True,
    include_decision_notes=True,
    generation_retries=2,
    retry_cooldown_seconds=20,
    evidence_token_budget=2400,
    selected_evidence_target_min_tokens=800,
    selected_evidence_target_max_tokens=1500,
    max_context_tokens=4096,
)
pipeline = Phase4RAGPipeline(config)
# Complete load(), chunk(), embed(), and index() before answering.
result = Phase4Runner(pipeline=pipeline, config=config).run(
    questions=["What exact control applies?"],
    run_mode="smoke",
)
print(result.paths.report_html)
```

Phase 4 now supports semi-adaptive answer sections. With
`adaptive_answer_sections=True`, the generation prompt selects only headings
that fit the question shape while retaining detailed synthesis, selected-
evidence grounding, inline citations, weak-evidence caveats, and explicit
evidence gaps. Set it to `False` to restore the previous fixed Phase 4 template
for reproducibility. Phase 5 now provides the separate full response planner;
no benchmark quality improvement is claimed from the Phase 4 prompt refinement.

For day-to-day Phase 4 runs, edit the clearly marked `USER CONFIGURATION`
section near the top of `scripts/run_phase4_batch.py`, especially
`QUESTIONS_FILE`, and then click **Run Python File** in VS Code. The same
zero-argument workflow works from PowerShell, an activated virtual environment,
or the explicit project interpreter:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py
# or, after activating the virtual environment:
python scripts\run_phase4_batch.py
```

The configuration section controls the question file, mode, answer-word limit,
semi-adaptive answer sections, generation retries/cooldown, reranker
device/batch/cache policy, and optional resume folder. Question lists are UTF-8
TXT files with one question per line or CSV files with a `question` column. For
a 440-question run, set `QUESTIONS_FILE` to that question list and optionally
adjust `MAX_ANSWER_WORDS`.

The script performs the notebook-equivalent load, chunk, embed, index, and
`Phase4Runner` sequence. Progress is flushed immediately, and no diagnostic,
dry-run, health-check, or configuration-resolution stage delays normal
execution.

CLI options remain available as advanced one-off overrides:

```powershell
python scripts\run_phase4_batch.py `
  --mode benchmark `
  --questions-file <path-to-benchmark-csv>
```

Terminal manual-QA runs process every loaded question by default. Only an
explicit `--max-questions N` truncates terminal input; `--large-run` remains an
accepted compatibility flag but is no longer required. The 25-question guard
is restricted to the interactive notebook workflow, where it protects the
kernel from rendering a large trace set. Smoke mode remains capped at three
questions, and benchmark behavior is unchanged.

For long local-model runs, Phase 4 retries retryable Ollama generation failures
without repeating retrieval or reranking. The defaults allow two retries after
the initial attempt with a 20-second cooldown. Every question attempt updates
`partial_results.csv`, `partial_results.jsonl`,
`partial_retrieval.jsonl`, and `checkpoint.json` inside the run folder.
Successful occurrences are keyed by original index plus normalized-question
hash, so duplicate question text resumes safely.

Optional advanced long-run override:

```powershell
python scripts/run_phase4_batch.py `
  --questions-file <path-to-question-file> `
  --max-answer-words <word-limit> `
  --generation-retries <retry-count> `
  --retry-cooldown-seconds <seconds>
```

Resume the same question file and options against the interrupted run folder:

```powershell
python scripts/run_phase4_batch.py `
  --questions-file <path-to-question-file> `
  --resume <path-to-run-folder> `
  --max-answer-words <word-limit> `
  --generation-retries <retry-count> `
  --retry-cooldown-seconds <seconds>
```

Completed questions are skipped; failed or interrupted occurrences are retried.
If retries remain exhausted, the final CSV retains a `generation_failed` row
with the original exception type/message while all standard reports are still
generated. For the normal edit-and-run workflow, set `RESUME_RUN_FOLDER` and
`MAX_ANSWER_WORDS` in `USER CONFIGURATION`; `--resume` and
`--max-answer-words` are optional CLI equivalents.

Reranking occurs after RRF because dense, BM25, and RRF scores are not
calibrated for direct averaging. The selector can enforce maximum evidence
count, reranker threshold, source diversity, redundancy reduction, and a
smaller evidence-token budget. Thresholding is advisory: if it would starve a
non-empty candidate pool, Phase 4 retains the configured evidence floor and
marks fallback chunks as weak/low-confidence. Normal QA targets roughly
800--1500 selected-evidence tokens rather than maximizing token reduction.
Selected chunks do not automatically make a question answerable. Extractive
fallback is limited to generator refusal with strong or mixed evidence whose
top and average reranker scores pass `min_fallback_reranker_score`; all-weak or
all-selection-fallback evidence is blocked by default. Questions that appear
to require live/current/external data, such as weather, share prices, IPL
results, cafeteria menus, or live network topology, return
`unsupported_query` unless indexed evidence directly supports the request.

Phase 4 answer statuses have distinct meanings:

- `answered`: a grounded generated answer, or a labeled extractive fallback
  backed by sufficient evidence.
- `insufficient_evidence`: indexed evidence is absent, weak, irrelevant, or
  otherwise below the fallback gate.
- `unsupported_query`: the request appears to need live/current/external data
  not directly supported by the indexed documents.
- `generation_failed`: local generation exhausted its retries and no response
  artifact could be completed.

Phase 4 bundles use
`outputs/batch_answers/04_Reranking_and_Evidence_Selection/run_<timestamp>/`
with the established Phase 3 artifact names.
The standalone Phase 4 HTML report turns recognized `[1]` and
`[source | Page N | Chunk ID]` answer markers into inline PDF-page citation
badges. Answers without markers receive compact citation chips, and the full
reference list remains collapsible. CSV/XLSX citation columns are unchanged.
The report header also provides Light, Dark, and System theme controls. System
mode follows the operating-system preference by default, while explicit
choices persist locally in the browser. All theme assets are embedded, keeping
the report standalone and double-click openable.

`CrossEncoderReranker` remains lazy: no model is loaded during pipeline
construction. On the first `answer()` call it always attempts
`local_files_only=True` first. If the model is cached, execution remains
offline. If the cache misses and `reranker_local_files_only=False`, the model is
downloaded and cached; later processes use that cache without code changes or
internet access. Strict offline deployments use:

```python
config = Phase4Config(reranker_local_files_only=True)
```

In strict mode a missing model fails with the configured model name, staging
instructions, and a reminder that `MockReranker` is available for automated
tests.

Weak reranker scores no longer mean “no evidence.” When usable chunks exist,
the selector falls back to ranked evidence, records `selection_reason` and
`evidence_confidence`, and the answer is labeled with a caution when all
selected evidence is weak. Zero selected chunks are reserved for retrieval with
no usable text. Discards use normalized reasons: `threshold_failed`,
`redundancy`, `source_diversity_limit`, `token_budget`, `empty_text`, and
`lower_rank_fallback`.

Phase 4 optimizes evidence precision, not answer brevity. Its generation prompt
retains the strict Phase 3 grounding and citation rules while requesting a
detailed, structured synthesis of operational implications, supported actions,
risks, gaps, caveats, and decision notes. `min_answer_words` is a target only
when evidence supports that depth; it never authorizes padding or unsupported
claims. The generator continues to receive only selected evidence.

The default token manager uses the configured local tiktoken encoding
(`cl100k_base` by default); it does not load or download a model. Injecting a
compatible encoder allows a future model-specific tokenizer without changing
context, evaluation, or reporting code. Set `max_context_tokens=None` to retain
the Phase 2 character-budget fitting path; token reporting remains exact
tiktoken output. Run bundles are written below
`outputs/batch_answers/03_Hybrid_Retrieval/run_<timestamp>/` and include
`results.csv`, `results.xlsx`, `report.html`, configuration, summary, retrieval,
metrics, logs, figures, and per-question context traces.

Reusable Phase 2 debugging helpers in `visualization.py` convert live pipeline
traces into pandas tables and matplotlib plots. They cover query variants,
single- versus multi-query retrieval, deduplication frequency, neighbor
provenance, score strength, source/page concentration, retrieval funnels,
character-based context compression, section balance, citation quality, batch
answer status, retrieval traces, and per-question latency. Notebook 02 only
supplies real pipeline outputs to these helpers.

Embedded Qdrant permits only one process per storage path. Close other notebook
kernels or clients before reopening the same `data/qdrant/` directory. Local
server mode supports concurrent clients and is recommended for large corpora.

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
  implementation state, limitations, and qualification boundaries.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) describes the long-term phase-by-phase
  architectural direction without treating planned capabilities as implemented.
# Phase 5: adaptive agentic response planning

Phase 5 is available as an optional, fully local validation and response
planning layer over Phase 4. It adds typed multimodal evidence, per-agent model
capability routing, bounded consensus/revision, and an offline Decision
Intelligence HTML dashboard. Phase 4 retrieval and evidence selection remain
unchanged. See [docs/phase5.md](docs/phase5.md) and
`notebooks/05_Agentic_Response_Planning.ipynb`.

For live local execution monitoring, run
`python -m cial_knowledge_os.live.command_center` and see
[the command-center guide](docs/phase5-live-command-center.md). The dashboard
uses only loopback FastAPI/SSE, packaged assets, and local device telemetry.
