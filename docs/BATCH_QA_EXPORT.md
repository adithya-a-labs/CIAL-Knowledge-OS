# Batch Question-Answer CSV Export

## Purpose

The batch export API keeps notebook cells small while producing repeatable,
inspectable evaluation artifacts. It uses the existing local pipeline to retrieve
evidence and generate each answer, then records citations, scores, model details,
and latency metrics in a versioned CSV.

No internet service, hosted model, cloud storage, or external API is used.

## Notebook Usage

Prepare and index the pipeline as usual, then pass the notebook's question list:

```python
from cial_knowledge_os.batch_qa import export_batch_answers

csv_path = export_batch_answers(
    pipeline=pipeline,
    questions=questions,
)

print(csv_path)
```

For `BasicRAGPipeline`, `export_batch_answers()` checks readiness before starting
the batch. If `load()`, `chunk()`, `embed()`, and `index()` have not completed,
it raises an actionable `RuntimeError` instead of writing one failed row per
question. Indexing remains explicit because it can rebuild local vector storage.

The public package also exports `export_batch_answers`, so this is equivalent:

```python
from cial_knowledge_os import export_batch_answers
```

An explicit experiment name and retrieval depth are optional:

```python
csv_path = export_batch_answers(
    pipeline=pipeline,
    questions=questions,
    run_name="01_Basic_RAG",
    top_k=5,
)
```

`top_k` applies only during the export and the pipeline's configured value is
restored afterward. It targets `top_k` for Phase 1 and `retrieval_top_k` for
Phase 2. Without `run_name`, the API infers a name from the pipeline class and
falls back to `batch_qa`.

Questions may alternatively be loaded from a UTF-8 text file with one question per
line or a CSV file containing a `question` column:

```python
csv_path = export_batch_answers(
    pipeline=pipeline,
    questions_path="data/eval/questions.csv",
)
```

## Output Structure and Versioning

The API creates the standard repository-local output tree when needed:

```text
outputs/
|-- batch_answers/
|-- evaluations/
|-- benchmarks/
|-- logs/
`-- exports/
```

Each run receives a sanitized experiment subdirectory and a monotonically
increasing filename:

```text
outputs/batch_answers/01_Basic_RAG/01_Basic_RAG-v1.csv
outputs/batch_answers/01_Basic_RAG/01_Basic_RAG-v2.csv
```

Files are created exclusively. Existing exports are never overwritten, including
when two local processes attempt to claim the same version.

The standard top-level output roots are `batch_answers/`, `benchmarks/`,
`evaluations/`, `exports/`, and `logs/`. Some may be empty until a workflow
creates an artifact. Future phases must extend this hierarchy rather than add a
new top-level `artifacts/` directory.

## CSV Schema

Columns are written in this order:

| Column | Meaning |
|---|---|
| `question` | Input question. |
| `answer` | Grounded answer returned by the pipeline. |
| `sources` | JSON array of source identifiers or paths. |
| `source_files` | JSON array of source file names. |
| `page_numbers` | JSON array of page numbers, including `null` when unavailable. |
| `chunk_ids` | JSON array of traceable chunk identifiers. |
| `retrieval_scores` | JSON array of retrieval scores in result order. |
| `top_k` | Retrieval depth requested for the batch. |
| `retrieved_chunks` | Number of chunks returned. |
| `answer_latency_seconds` | Local answer-generation duration. |
| `retrieval_latency_seconds` | Local retrieval duration. |
| `total_latency_seconds` | Full duration for the question, including failures. |
| `model_name` | Configured local generation model, when available. |
| `embedding_model` | Configured local embedding model, when available. |
| `timestamp` | Timezone-aware ISO timestamp for the row. |
| `status` | `success` or `failed`. |
| `error` | Exception message for a failed row; blank on success. |

List-like citation fields are compact JSON arrays inside CSV cells. Files use
UTF-8 with a byte-order mark for compatibility with Excel and LibreOffice.

If one question fails, its row has `status` set to `failed` and contains the
exception message in `error`; remaining questions continue normally.

## Phase 2 Extension

Passing a `Phase2RAGPipeline` reuses the same exporter and calls its complete
`answer()` workflow for every question. Query transformations, multi-query
retrieval, deduplication, neighbor expansion, context construction, local
generation, and citation formatting are therefore reflected in each row.

The original Phase 1 columns above remain unchanged. Phase 2 exports append:

| Column | Meaning |
|---|---|
| `query_variants` | JSON array containing each transformation technique and query. |
| `chunks_before_deduplication` | Combined chunks retrieved across all query variants. |
| `chunks_after_deduplication` | Unique chunks after `(source, page, chunk_id)` deduplication. |
| `chunks_after_neighbor_expansion` | Evidence count after adding configured neighbors. |
| `merged_context_sections` | Contiguous sections produced by overlap merging. |
| `final_context_sections` | Merged sections retained after context compression. |
| `final_context_characters` | Exact final prompt-context length in characters. |
| `final_context_tokens_estimate` | Exact centralized tiktoken count; the legacy column name is preserved for compatibility. |
| `answer_status` | `Answered` or `Insufficient Evidence`; separate from export success/failure. |
| `retrieval_trace` | Concise query-to-context audit trail for the row. |

The existing `status` column continues to represent export execution
(`success` or `failed`). `answer_status` records whether the corpus supported a
grounded answer. Existing source, page, chunk, and score columns use the final
compressed evidence blocks for Phase 2 so they align with answer citations.

Phase 3 is expected to preserve this external CSV behavior while adding an
isolated per-run bundle with CSV, XLSX, standalone HTML, configuration,
retrieval, metrics, logs, figures, and context artifacts.

## Phase 3 Extension

`collect_batch_answers()` is the shared collection path used by the legacy CSV
export and `Phase3Runner`. Existing Phase 1 and Phase 2 columns remain in their
original order. Phase 3 appends:

| Column | Meaning |
|---|---|
| `retrieval_mode` | `dense`, `bm25`, or `hybrid`. |
| `dense_top_k` | Dense candidate depth. |
| `bm25_top_k` | Lexical candidate depth. |
| `rrf_k` | RRF rank constant. |
| `final_context_tokens` | Configured-tokenizer context usage. |
| `context_budget` | Effective token or character limit. |
| `context_budget_type` | `tokens` or backward-compatible `characters_legacy`. |
| `token_encoding` | Configured tiktoken encoding or injected tokenizer name. |
| `pdf_links` | JSON array of clickable evidence links. |
| `retrieval_sources` | JSON array of contributing retriever names. |
| `dense_result_count` | Raw dense candidates collected across query variants. |
| `bm25_result_count` | Raw BM25 candidates collected across query variants. |
| `fused_result_count` | RRF candidates collected across query variants. |
| `final_context_chunk_count` | Evidence sections retained in final context. |
| `context_tokens_used` | Exact configured-tokenizer context usage. |
| `token_utilization` | Percentage of the configured token budget used. |
| `generation_latency_seconds` | Local generation latency for the question. |
| `citation_count` | Structured citations attached to the answer. |
| `unique_source_count` | Unique documents represented in final context. |

In hybrid rows the legacy `retrieval_scores` column contains fused RRF scores,
not cosine similarities. Retriever-specific raw scores and ranks remain in the
full response and `retrieval.json` trace.

`Phase3Runner` writes the configured non-overwriting bundle below
`outputs/batch_answers/03_Hybrid_Retrieval/run_<timestamp>/`. The XLSX workbook
formats the established columns and makes the first PDF citation clickable.
The HTML report embeds all styles and data and requires no server or external
assets.

`retrieval.json` contains the full per-question execution trace: query variants,
raw dense and BM25 candidates, RRF ranks, overlap, deduplication and neighbor
statistics, the context funnel, exact token usage, generation and artifact
latencies, citations, source diversity, artifact paths, and decision-focused
recommendations. The standalone HTML renders the same trace in collapsible
offline sections; CSV and XLSX retain only compact summary fields.

## Phase 4 Extension

Phase 4 uses embedded Qdrant by default. For large local corpora, start the
on-premises Docker service with
`docker compose -f docker-compose.qdrant.yml up -d`, verify
`curl http://localhost:6333/healthz`, and construct `Phase4Config` with
`qdrant_mode="server"` and `qdrant_url="http://localhost:6333"`. This does not
use Qdrant Cloud. Existing embedded points can be copied with
`scripts/migrate_embedded_qdrant_to_server.py`; see the repository README for
the complete command and overwrite safeguards. Server indexing defaults to
`qdrant_batch_size=32` to bound Qdrant client's JSON serialization memory;
`qdrant_upsert_wait=True` retains synchronous upsert behavior. Keep batches
small for large vectors or metadata-rich points.

`Phase4Runner` reuses `collect_batch_answers()`, the Phase 3 columns, and the
same `RunManager`. It appends these machine-readable columns:

Large Phase 4 runs should use `scripts/run_phase4_batch.py` from a terminal
rather than relying on a long-lived Jupyter kernel. The script performs the
same load, chunk, embed, index, and `Phase4Runner` sequence as the notebook,
loads configured CSV/TXT question inputs, and does not render traces inline.
The clearly marked `USER CONFIGURATION` section at the top of the script is the
day-to-day interface. Set `QUESTIONS_FILE` and any mode, retry, answer-length,
reranker, or resume values there, then click **Run Python File** in VS Code.
The same zero-argument run works from PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase4_batch.py
# or from an activated virtual environment
python scripts\run_phase4_batch.py
```

TXT inputs contain one question per line, and CSV inputs must contain a
`question` column. For a 440-question run, change `QUESTIONS_FILE` and
optionally `MAX_ANSWER_WORDS`.

Indexing is incremental by default. Set `FORCE_REBUILD_INDEX = True` in the
user configuration or pass `--force-rebuild-index` for a one-off safe rebuild.
The pipeline writes `data/indexes/document_manifest.json`; run `summary.json`
and `metrics.json` include an additive `indexing_summary` with file
classification, chunk updates, BM25/vector update flags, reused chunks, and the
embedding-time-saved estimate.

CLI options are advanced, one-off overrides; none are required:

```powershell
# Small execution check
python scripts\run_phase4_batch.py --mode smoke --questions-file <path-to-question-file>

# Benchmark input override
python scripts\run_phase4_batch.py --mode benchmark --questions-file <path-to-benchmark-csv>
```

`--mode` supports `smoke`, `manual_qa`, and `benchmark`. Additional controls
include `--max-questions`, `--max-answer-words`, retry/cooldown overrides,
`--reranker-device`, `--reranker-batch-size`, `--local-files-only`,
`--force-rebuild-index`, and `--resume`. The existing runner remains responsible for CSV, XLSX,
standalone HTML, configuration, summary, metrics, retrieval traces, logs,
per-question context, and supported SVG/HTML visualization exports.

The terminal runner is unbounded for manual QA: all questions loaded from the
resolved default or CLI-provided file are processed unless `--max-questions`
is supplied.
The legacy `--large-run` flag remains accepted for command compatibility but is
not required. The notebook retains its separate 25-question safety limit and
warning, smoke mode retains its three-question limit, and benchmark mode is
unchanged. Terminal output and structured logs record the loaded count, counts
entering Phase 4/Phase 3 and batch execution, and the rows written to
`results.csv`.

### Long-run reliability and resume

Phase 4 retries only the local generation call when Ollama reports runner,
stream, transport, HTTP 500, or allocation failures. Retrieval, RRF, reranking,
and evidence selection are not repeated. `generation_retries=2` means up to
three total generation attempts; `retry_cooldown_seconds=20` controls the pause
between retryable failures. Exhausted attempts produce a final row with
`answer_status=generation_failed` and the original error type/message.

Phase 4 keeps answer outcome separate from export success:

- `Answered` is a grounded generated answer or a sufficient-evidence
  extractive fallback after generator refusal.
- `Insufficient Evidence` means evidence was absent, weak, irrelevant, or
  below the configured fallback score gate.
- `Unsupported Query` means the request appears to require live/current/
  external data that the indexed documents do not directly support.
- `generation_failed` means local generation exhausted retries and the
  question row failed.

CSV and XLSX retain these values in `answer_status`; the standalone HTML
report renders a distinct status badge. Aggregate metrics include
`unsupported_query_count`, `insufficient_evidence_count`,
`extractive_fallback_count`, and `fallback_blocked_count`.

Each attempted question immediately updates:

```text
partial_results.csv
partial_results.jsonl
partial_retrieval.jsonl
checkpoint.json
```

`checkpoint.json` contains the run path/id, indexed normalized-question hashes,
completed and failed occurrences, last completed index, configuration snapshot,
and update timestamp. Identity combines original index and question hash, so
duplicate text is not incorrectly skipped.

```powershell
# Long run with bounded answer generation
python scripts/run_phase4_batch.py `
  --questions-file <path-to-question-file> `
  --max-answer-words <word-limit> `
  --generation-retries <retry-count> `
  --retry-cooldown-seconds <seconds>

# Resume after interruption using the same question input/order
python scripts/run_phase4_batch.py `
  --questions-file <path-to-question-file> `
  --resume <path-to-run-folder>
```

Resume reconstructs final artifacts from prior successful checkpoints and new
attempts. Previously failed occurrences remain eligible for another attempt.
The question file and `--max-questions` setting must match the original run.

Treat question lists as reviewable evaluation data. Update the input file and
point `QUESTIONS_FILE` at it for normal use; `--questions-file` can select any
supported CSV or TXT input for a one-off run without changing the output
contract.

| Column | Meaning |
|---|---|
| `candidate_chunk_count` | Post-RRF, deduplicated candidates eligible for reranking. |
| `reranked_candidate_count` | Candidates scored and ordered by the local reranker. |
| `selected_chunk_count` | Evidence chunks retained by all enabled selection strategies. |
| `discarded_chunk_count` | Candidates removed before context construction. |
| `candidate_tokens` | Exact tokens in the Phase 3-style serialized candidate context. |
| `selected_evidence_tokens` | Exact tokens in selected chunk text before final formatting. |
| `token_reduction_percent` | Candidate-to-final-context token reduction. |
| `average_reranker_score` | Mean selected-evidence reranker score; model-specific, not universally calibrated. |
| `strong_evidence_count` | Selected chunks at or above the configured strong threshold. |
| `medium_evidence_count` | Selected chunks between configured medium and strong thresholds. |
| `weak_evidence_count` | Selected chunks below the configured medium threshold. |
| `reranker_latency_seconds` | Local cross-encoder scoring and sorting latency. |
| `evidence_selection_latency_seconds` | Keep/discard decision latency. |
| `usable_candidate_count` | Non-empty candidates eligible for normal or fallback selection. |
| `threshold_pass_count` | Candidates meeting the configured reranker threshold. |
| `fallback_used` | Whether adaptive selection added evidence below the threshold. |
| `evidence_confidence` | `strong`, `mixed`, `weak`, or `none`. |

Run bundles are written below:

```text
outputs/batch_answers/04_Reranking_and_Evidence_Selection/run_<timestamp>/
|-- results.csv
|-- results.xlsx
|-- report.html
|-- config.json
|-- summary.json
|-- metrics.json
|-- retrieval.json
|-- logs.txt
|-- figures/
`-- context/
```

`retrieval.json` contains the full Phase 3 trace plus original RRF rank,
reranker score/rank, selected/discarded status, discard reason, evidence
strength, metadata completeness, citation availability/link, token counts,
final-context inclusion, latency, answer, citations, and artifact paths.

The selector defaults to `min_selected_evidence=3`,
`max_selected_evidence=8`, `fallback_top_n=3`, and an 800--1500 selected-token
target. A threshold miss does not erase all usable evidence: ranked fallback
chunks are retained and marked weak. Zero selection is reserved for no
candidates or candidates with empty text. Exact discard reasons are
`threshold_failed`, `redundancy`, `source_diversity_limit`, `token_budget`,
`empty_text`, and `lower_rank_fallback`; aggregate counts are included in
`metrics.json` and `report.html`.

The Phase 4 HTML report remains standalone and offline. It adds Executive
Summary, Answers, Citations, Reranking Trace, Evidence Selection, Token
Reduction, Latency Breakdown, Evidence Quality, Enterprise File Format
Readiness, OCR Processing Summary, Source Diversity, Selected versus
Discarded, Discard Reasons, comparison-status, and collapsible context/debug
sections. Its charts are inline SVG; no CDN or external JavaScript is required.

Phase 4 run bundles also include file-format readiness exports:
`file_format_summary.csv`, `file_extension_distribution.csv`, and
`skipped_files.csv`. The XLSX workbook keeps the existing active results sheet
and adds matching `file_format_summary`, `file_extension_distribution`, and
`skipped_files` sheets. These outputs are generated from the central Enterprise
File Format Registry, where only `SUPPORTED_NOW` and `OCR_SUPPORTED` files are
processable; `RECOGNIZED_FUTURE_SUPPORT` and `UNSUPPORTED` files are reported
but not ingested.

The report includes an accessible Light/Dark/System theme control in its
header. System mode is the default and follows `prefers-color-scheme`; an
explicit choice is stored in browser `localStorage` and restored when the
double-click-openable report is reopened. Both palettes cover answer cards,
metrics, tables, citations and hover previews, status badges, debug details,
trace/code blocks, and inline SVG containers. Theme CSS and JavaScript remain
embedded in `report.html`, so no server, CDN, remote font, or network access is
required. If JavaScript is disabled, the report retains a complete readable
light theme.

Answer cards render the complete generated Markdown without preview truncation.
Recognized numeric markers such as `[1]` and structured
`[source | Page N | Chunk ID]` markers become inline PDF-page links with
source/page/chunk/score hover details. When the model omits inline markers, the
answer receives compact citation chips. The complete reference list remains
available in a collapsible citation-details section. Unsafe or unavailable
links are rendered as non-clickable labels; CSV/XLSX citation fields are not
changed.
Phase 4 uses the Phase 3 grounding and citation contract but requests detailed,
structured decision support from selected evidence only. Phase 4 now supports
semi-adaptive answer sections: with `adaptive_answer_sections=True`, the prompt
chooses only question-relevant section families; `False` restores the previous
fixed template for reproducibility. This is not a full response planner; full
adaptive response planning is deferred to Phase 5. The evidence selector
reduces irrelevant context; it does not intentionally shorten answers. No
benchmark quality improvement is claimed from this prompt-only change.

The implementation supports `smoke`, `manual_qa`, `benchmark`, and
`export_only` modes and `compact`/`full` traces. Full benchmark qualification is
pending; exported token and score diagnostics must not be described as proven
quality improvements without benchmark evidence.

Reranker model loading is lazy and cache-first. With the developer default
`reranker_local_files_only=False`, the first answer downloads and caches a
missing configured model; later batch runs load it locally. Set
`reranker_local_files_only=True` for enterprise offline runs. In that mode a
cache miss skips download and fails before export with manual staging guidance.
