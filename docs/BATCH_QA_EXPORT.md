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
restored afterward. Without `run_name`, the API infers a name from the pipeline
class and falls back to `batch_qa`.

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
