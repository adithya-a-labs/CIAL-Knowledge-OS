# Automated Offline Evaluation

The reusable evaluation framework lives in `src/cial_knowledge_os/` and is not
tied to a notebook phase. It currently evaluates the frozen Phase 2 dense
baseline and is designed to accept later pipeline implementations.

## Modules and Responsibilities

| Module | Responsibility |
|---|---|
| `benchmark_loader.py` | Load benchmark questions and metadata into typed records. |
| `evaluation_metrics.py` | Score individual answers, aggregate metrics, and rank configurations using deterministic offline heuristics. |
| `evaluation_report.py` | Build recommendations and write the Markdown recommendation report. |
| `experiment_config.py` | Define immutable configurations and deterministic parameter grids. |
| `experiment_runner.py` | Execute sweeps, isolate question failures, and write experiment, summary, recommendation, and dashboard artifacts. |
| `visualization_dashboard.py` | Generate a self-contained offline HTML dashboard from embedded CSV and summary data. |

`visualization.py` provides notebook-oriented pandas and matplotlib diagnostics;
`batch_qa.py` provides general batch answer exports without ground-truth scoring.

## Artifact Contract

A sweep writes this artifact contract:

```text
outputs/batch_answers/<phase>/
├── experiments/
│   └── experiment_001.csv
├── summary/
│   └── experiment_summary.csv
└── reports/
    ├── recommendation.md
    └── dashboard.html
```

`dashboard.html` contains the generated CSV and summary data as an embedded
snapshot. This is intentional: browsers commonly block a local `file://` page
from fetching neighboring CSV files. Embedding makes the report self-contained,
offline, and portable. Re-running a sweep refreshes the snapshot automatically.

## Minimal usage

```python
from pathlib import Path

from cial_knowledge_os import (
    ExperimentGrid,
    ExperimentRunner,
    ReconfiguringPipelineFactory,
    load_benchmark,
)

root = Path.cwd()
benchmark = load_benchmark(
    root / "data/benchmarks/cisg/benchmark_answers.csv",
    metadata_path=root / "data/benchmarks/cisg/benchmark_metadata.json",
)

# `pipeline` is an already loaded, embedded, and indexed local pipeline.
runner = ExperimentRunner(
    pipeline_factory=ReconfiguringPipelineFactory(pipeline),
    benchmark=benchmark,
    output_root=root / (
        "outputs/batch_answers/"
        "02_Query_Transformations_and_Context_Construction"
    ),
)
result = runner.run(
    ExperimentGrid(
        {
            "retrieval_top_k": [3, 5, 10, 15, 20],
            "max_context_chars": [3000, 6000, 12000, 20000],
            "neighbor_window": [0, 1, 2],
            "enable_multi_query": [True, False],
            "enable_neighbor_expansion": [True, False],
        }
    )
)
print(result.dashboard_file)
```

The adapter restores the pipeline's original configuration after the sweep and
reuses its index, avoiding repeated document loading and embedding.

## Extension contract

- Add grid parameters without changing the runner. They are exported as
  `config_<name>` columns.
- Add future evaluation fields through `metric_hooks`; returned keys become CSV
  columns and are available to downstream report code.
- Implement another pipeline through the small `answer(question)` protocol and
  expose `config` plus an optional `metrics` mapping.
- Keep standard metric names where possible:
  `total_latency`, `retrieval_latency`, `context_construction_latency`,
  `generation_latency`, `citation_quality`, and `hallucination_rate`.
- Add new dashboard panels by consuming embedded row or summary columns. The
  existing artifact schema remains valid.

## Evaluation behavior

Scoring is deterministic and offline. Supported questions pass when the answer
is non-empty, is not a safe failure, and meets the keyword coverage threshold.
Unsupported questions pass only when the pipeline safely refuses. Forbidden
keywords and unsafe answers to unsupported questions contribute to the
hallucination metric. This is a heuristic benchmark, not semantic entailment;
a future local evaluator can be added as a metric hook.

The framework does not currently calculate semantic entailment, labeled
retrieval recall, BM25/vector contribution, Reciprocal Rank Fusion quality, or
reranker relevance. Those metrics must be added only when the corresponding
retrieval stages exist.

## Benchmark

The current frozen benchmark lives at:

```text
data/benchmarks/cisg/
|-- benchmark_answers.csv
|-- benchmark_metadata.json
|-- cisg_questions_v1.txt
|-- README.md
`-- CHANGELOG.md
```

It contains 200 questions. Do not modify the frozen files for a new phase; use
the same version for Phase 2 versus Phase 3 comparisons or publish a separately
versioned benchmark.

## Phase 3 Extension

Phase 3 reuses the runner and benchmark contracts. `Phase3RAGPipeline` exposes
the same `answer(question)`, configuration, metrics, context-stage, and citation
surfaces as Phase 2, while additive evaluation columns record retrieval mode,
dense/BM25 depths, RRF configuration, and exact centralized tiktoken context
usage. The legacy `estimated_tokens` column remains for schema compatibility but
contains an exact token-manager count.

Use `ReconfiguringPipelineFactory` with `retrieval_mode` values `dense` and
`hybrid` to compare both modes against the same frozen benchmark while reusing
loaded documents, embeddings, Qdrant state, and the BM25 index. The factory
invokes the pipeline configuration hook after changes and restores the original
configuration after the sweep.

`Phase3Runner` separately creates the isolated per-run artifact bundle. It is a
run report, not a replacement for the aggregate evaluation dashboard.

## Operational considerations

The full example grid contains 240 configurations and 48,000 generation calls
for a 200-question benchmark. Run a small smoke grid first. Embedding and index
construction should happen once before the sweep; generation remains the main
expected bottleneck. Experiment CSVs are written after each configuration so
completed work remains inspectable if a later configuration fails.
