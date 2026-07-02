# Automated Offline Evaluation

The reusable evaluation framework lives in `src/cial_knowledge_os/` and is not
tied to a notebook phase. A sweep writes this artifact contract:

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

## Operational considerations

The full example grid contains 240 configurations and 48,000 generation calls
for a 200-question benchmark. Run a small smoke grid first. Embedding and index
construction should happen once before the sweep; generation remains the main
expected bottleneck. Experiment CSVs are written after each configuration so
completed work remains inspectable if a later configuration fails.
