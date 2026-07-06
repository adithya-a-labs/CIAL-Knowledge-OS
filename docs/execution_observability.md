# Execution & Observability Framework

The Execution & Observability Framework (EOF) is the shared, local-only
instrumentation layer for CIAL Knowledge OS. Pipelines perform work; EOF
observes that work. EOF does not choose retrieval methods, retry work, change
answers, select evidence, or control agent decisions.

## Architecture

Pipeline and runner boundaries emit typed `ExecutionEvent` objects to an
in-process `EventBus`. Subscribers independently maintain progress, aggregate
timings, render optional console output, collect local machine telemetry, and
write durable run files. Subscriber failures are isolated and cannot interrupt
pipeline execution.

The package is under `src/cial_knowledge_os/execution/` and is phase-neutral.
Current integration covers Phase 4 question and stage execution, shared
indexing, infrastructure preflight, and Phase 5 agent events. Future phases
should emit the same lifecycle events instead of creating separate monitoring
paths.

Typical question lifecycle:

```text
run_started
  question_started
    retrieval_started -> retrieval_completed
    reranking_started -> reranking_completed
    evidence_selection_started -> evidence_selection_completed
    generation_started -> generation_completed | generation_failed
  question_completed | question_failed
  checkpoint_written
  ...
  export_started -> export_completed
batch_completed
run_completed | run_failed
```

Phase 5 adds `agent_started`, `agent_completed`/`agent_failed`, and
`consensus_decided`. Indexing adds `indexing_started`, per-batch
`indexing_progress`, health events, and `indexing_completed`/`indexing_failed`.

## Run files

Each observed run writes to:

```text
outputs/runs/<run_id>/
|-- execution_trace.jsonl
|-- progress.json
`-- progress.log
```

- `execution_trace.jsonl` contains one complete event per line.
- `progress.json` is the latest machine-readable snapshot, including counters,
  ETA, status distribution, current stage/question, warnings, errors, metrics,
  and telemetry.
- `progress.log` is an append-only, human-readable lifecycle log.

These files are additive. Existing phase CSV, XLSX, HTML, JSON, checkpoint, and
resume contracts are unchanged.

Inspect a run while it is active:

```powershell
type outputs\runs\<run_id>\progress.log
type outputs\runs\<run_id>\progress.json
```

## Console and telemetry

When `rich` is installed and enabled, EOF uses a compact Rich renderer.
Otherwise it automatically uses plain terminal lines. `psutil` enables CPU,
RAM, disk, and process-memory fields. If `psutil` is absent, those fields are
omitted. GPU telemetry uses local `nvidia-smi` when available. No telemetry is
sent over the network.

## Configuration

All options are fields on `KnowledgeOSConfig` and its phase subclasses:

```python
config = Phase4Config(
    observability_enabled=True,
    observability_console=True,
    observability_rich="auto",
    observability_trace_jsonl=True,
    observability_progress_log=True,
    observability_telemetry=True,
    observability_telemetry_interval_seconds=5,
    observability_console_refresh_seconds=1,
    observability_output_dir="outputs/runs",
)
```

Set `observability_enabled=False` to use the safe no-op path. Individual
renderers and writers can also be disabled without changing pipeline behavior.

The normal Phase 4 command enables EOF from the resolved configuration:

```powershell
python scripts/run_phase4_batch.py
```

Programmatic callers may create and pass one run-scoped manager:

```python
manager = ExecutionManager.from_config(
    config, phase="Phase 4", run_mode="manual_qa"
)
pipeline.execution_manager = manager
runner = Phase4Runner(
    pipeline=pipeline,
    config=config,
    execution_manager=manager,
)
```

## Live command center

A future Jarvis-style command center should consume the same `EventBus`,
JSONL trace, or `progress.json`. It must remain a subscriber. It should not add
new instrumentation inside retrieval or generation and must never become the
component that decides pipeline work.
