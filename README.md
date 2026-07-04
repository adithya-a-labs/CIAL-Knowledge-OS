
# CIAL Knowledge OS

An enterprise-grade, fully offline, notebook-first RAG platform for enterprise
documentation. The current repository provides completed dense-retrieval Phase
1 and query/context-construction Phase 2 baselines plus implemented Phase 3
hybrid retrieval and Phase 4 reranking/evidence-selection architectures. Phase
3 and Phase 4 still await full local benchmark qualification. Agentic workflows,
access control, multimodal retrieval, contradiction detection, and production
applications remain target capabilities, not current implementation claims.

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
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_local_files_only=False,  # cache first; download once if missing
    reranker_batch_size=16,
    reranker_candidate_top_k=30,
    min_selected_evidence=3,
    max_selected_evidence=8,
    reranker_score_threshold=-4.0,
    fallback_to_top_n_if_empty=True,
    fallback_top_n=3,
    weak_evidence_answer_allowed=True,
    answer_detail_level="detailed",
    min_answer_words=250,
    max_answer_words=None,
    prefer_structured_answers=True,
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

For batch runs, use the terminal entry point so model execution is not tied to
a Jupyter kernel:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py --questions-file data/manual_qa/cybersecurity_questions.txt
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py --mode smoke --questions-file data/manual_qa/smoke_questions.txt
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py --mode benchmark
```

The default manual and smoke input is the version-controlled
`data/manual_qa/phase4_questions.txt`. Question lists are UTF-8 text files with
one question per line; CSV files with a `question` column remain supported via
`--questions-file`. Starter lists are provided for smoke checks, cybersecurity,
and airport operations under `data/manual_qa/`. The benchmark mode continues to
use the configured benchmark dataset when no override is supplied.

The script mirrors the notebook's Phase 4 initialization, renders no inline
traces, and prints the paths to the complete run bundle and existing SVG/HTML
visualizations. Use `--max-questions`, `--reranker-device`,
`--reranker-batch-size`, and `--local-files-only` as needed. Runs remain under
`outputs/batch_answers/04_Reranking_and_Evidence_Selection/run_<timestamp>/`.

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

Recommended 440-question run:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py `
  --questions-file data/manual_qa/phase4_questions.txt `
  --max-answer-words 450 `
  --generation-retries 2 `
  --retry-cooldown-seconds 20
```

Resume the same question file and options against the interrupted run folder:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py `
  --questions-file data/manual_qa/phase4_questions.txt `
  --resume outputs/batch_answers/04_Reranking_and_Evidence_Selection/run_<timestamp> `
  --max-answer-words 450 `
  --generation-retries 2 `
  --retry-cooldown-seconds 20
```

Completed questions are skipped; failed or interrupted occurrences are retried.
If retries remain exhausted, the final CSV retains a `generation_failed` row
with the original exception type/message while all standard reports are still
generated. `--max-answer-words` adds a grounded prompt upper bound; omitting it
preserves the existing detailed answer behavior.

Reranking occurs after RRF because dense, BM25, and RRF scores are not
calibrated for direct averaging. The selector can enforce maximum evidence
count, reranker threshold, source diversity, redundancy reduction, and a
smaller evidence-token budget. Thresholding is advisory: if it would starve a
non-empty candidate pool, Phase 4 retains the configured evidence floor and
marks fallback chunks as weak/low-confidence. Normal QA targets roughly
800--1500 selected-evidence tokens rather than maximizing token reduction.
Phase 4 bundles use
`outputs/batch_answers/04_Reranking_and_Evidence_Selection/run_<timestamp>/`
with the established Phase 3 artifact names.
The standalone Phase 4 HTML report turns recognized `[1]` and
`[source | Page N | Chunk ID]` answer markers into inline PDF-page citation
badges. Answers without markers receive compact citation chips, and the full
reference list remains collapsible. CSV/XLSX citation columns are unchanged.

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
  implementation state, limitations, and qualification boundaries.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) describes the long-term phase-by-phase
  architectural direction without treating planned capabilities as implemented.
