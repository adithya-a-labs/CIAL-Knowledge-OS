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
| `final_context_tokens_estimate` | Offline tokenizer-independent token estimate. |
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
| `context_budget_type` | `tokens` or backward-compatible `characters`. |
| `pdf_links` | JSON array of clickable evidence links. |
| `retrieval_sources` | JSON array of contributing retriever names. |

In hybrid rows the legacy `retrieval_scores` column contains fused RRF scores,
not cosine similarities. Retriever-specific raw scores and ranks remain in the
full response and `retrieval.json` trace.

`Phase3Runner` writes the configured non-overwriting bundle below
`outputs/batch_answers/03_Hybrid_Retrieval/run_<timestamp>/`. The XLSX workbook
formats the established columns and makes the first PDF citation clickable.
The HTML report embeds all styles and data and requires no server or external
assets.
