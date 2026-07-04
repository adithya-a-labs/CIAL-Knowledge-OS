"""Run Phase 4 batch QA with transparent, configuration-driven startup."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class StartupReporter:
    """Write timestamped, immediately visible CLI status messages."""

    def __init__(self, *, verbose: bool = False) -> None:
        self.started = perf_counter()
        self.verbose = verbose

    @property
    def elapsed(self) -> float:
        return perf_counter() - self.started

    def step(self, message: str) -> None:
        print(f"[{self.elapsed:8.2f}s] {message}", flush=True)

    def detail(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)


def _enable_immediate_output() -> None:
    """Request line-buffered, write-through output when the stream supports it."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except (OSError, ValueError):
                pass


def _print(message: str = "") -> None:
    print(message, flush=True)


def positive_integer(value: str) -> int:
    """Parse a strictly positive CLI integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return parsed


def non_negative_integer(value: str) -> int:
    """Parse a non-negative CLI integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def non_negative_number(value: str) -> float:
    """Parse a non-negative CLI number."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def resolve_path(path: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a user/configuration path relative to the effective project root."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def load_questions(
    path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    """Load non-empty questions from CSV or line-oriented TXT input."""

    source = resolve_path(path, project_root=project_root)
    if not source.is_file():
        raise FileNotFoundError(f"Questions file not found: {source}")

    if source.suffix.casefold() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "question" not in reader.fieldnames:
                raise ValueError("Questions CSV must contain a 'question' column.")
            questions = [
                str(row.get("question") or "").strip()
                for row in reader
                if str(row.get("question") or "").strip()
            ]
    elif source.suffix.casefold() == ".txt":
        questions = [
            line.strip()
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("Questions file must use a .csv or .txt extension.")

    if not questions:
        raise ValueError(f"Questions file contains no usable questions: {source}")
    return questions


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 4 terminal argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 4 outside Jupyter and export the complete batch QA bundle."
        )
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional JSON object containing Phase4Config field values.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "manual_qa", "benchmark"),
        help="Execution mode; defaults to the resolved Phase4Config value.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="A .csv with a question column or .txt with one question per line.",
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--sample-documents-dir", type=Path)
    parser.add_argument("--raw-documents-dir", type=Path)
    parser.add_argument("--pdf-documents-dir", type=Path)
    parser.add_argument("--vector-db-dir", type=Path)
    parser.add_argument("--vector-collection")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--benchmark-metadata-file", type=Path)
    parser.add_argument("--embedding-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--reranker-model")
    parser.add_argument(
        "--large-run",
        action="store_true",
        help=(
            "Compatibility flag; terminal runs are already unbounded. "
            "Use --max-questions to set an explicit limit."
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=positive_integer,
        help="Run at most the first N questions from the selected source.",
    )
    parser.add_argument(
        "--reranker-device",
        choices=("cpu", "cuda", "auto"),
        help="Cross-encoder execution device.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=positive_integer,
        help="Cross-encoder scoring batch size.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require the configured reranker to exist in the local cache.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted Phase 4 run from its run folder.",
    )
    parser.add_argument(
        "--generation-retries",
        type=non_negative_integer,
        help="Generation retries after the initial attempt.",
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=non_negative_number,
        help="Cooldown between retryable generation attempts.",
    )
    parser.add_argument(
        "--max-answer-words",
        type=positive_integer,
        help="Optional upper word target for each generated answer.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved configuration and detailed startup settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview the run without initializing the pipeline.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check configured local dependencies and exit without running QA.",
    )
    return parser


def _load_config_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration file must contain a JSON object.")
    return dict(value)


def build_config(args: argparse.Namespace) -> Any:
    """Build Phase4Config from optional JSON plus explicit CLI overrides."""

    from cial_knowledge_os.config import Phase4Config, RunArtifactNames

    values = _load_config_values(args.config_file)
    valid_fields = {item.name for item in fields(Phase4Config) if item.init}
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(
            "Unknown Phase4Config field(s): " + ", ".join(unknown)
        )
    if isinstance(values.get("artifact_names"), dict):
        values["artifact_names"] = RunArtifactNames(**values["artifact_names"])
    if isinstance(values.get("evidence_selection_strategies"), list):
        values["evidence_selection_strategies"] = tuple(
            values["evidence_selection_strategies"]
        )

    values["project_root"] = args.project_root or values.get(
        "project_root",
        PROJECT_ROOT,
    )
    overrides = {
        "sample_data_dir": args.sample_documents_dir,
        "raw_data_dir": args.raw_documents_dir,
        "pdf_data_dir": args.pdf_documents_dir,
        "qdrant_dir": args.vector_db_dir,
        "qdrant_collection_name": args.vector_collection,
        "output_root": args.output_dir,
        "benchmark_csv_path": args.benchmark_file,
        "benchmark_metadata_path": args.benchmark_metadata_file,
        "embedding_model_name": args.embedding_model,
        "ollama_model_name": args.llm_model,
        "reranker_model_name": args.reranker_model,
        "reranker_device": args.reranker_device,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_local_files_only": args.local_files_only,
        "generation_retries": args.generation_retries,
        "retry_cooldown_seconds": args.retry_cooldown_seconds,
        "max_answer_words": args.max_answer_words,
        "phase4_run_mode": args.mode,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    # The terminal surface remains intentionally unbounded. Notebook callers
    # retain the existing interactive guard through their own config instance.
    values["allow_large_run"] = True
    return Phase4Config(**values)


def select_inputs(
    args: argparse.Namespace,
    config: Any,
) -> tuple[list[str], Any | None, str]:
    """Resolve questions from the actual CLI argument or benchmark config."""

    from cial_knowledge_os.benchmark_loader import Benchmark, load_benchmark

    benchmark: Benchmark | None = None
    mode = args.mode or config.phase4_run_mode

    if args.questions_file is not None:
        source = resolve_path(
            args.questions_file,
            project_root=config.project_root,
        )
        if mode == "benchmark" and source.suffix.casefold() == ".csv":
            benchmark = load_benchmark(source)
            questions = [item.question for item in benchmark.questions]
        else:
            questions = load_questions(
                source,
                project_root=config.project_root,
            )
    elif mode == "benchmark":
        source = config.benchmark_csv_path
        metadata_path = (
            config.benchmark_metadata_path
            if config.benchmark_metadata_path.is_file()
            else None
        )
        benchmark = load_benchmark(source, metadata_path=metadata_path)
        questions = [item.question for item in benchmark.questions]
    else:
        raise ValueError(
            f"{mode} mode requires --questions-file <path>. "
            "No question filename is assumed."
        )

    if args.max_questions is not None:
        questions = questions[: args.max_questions]
        if benchmark is not None:
            benchmark = Benchmark(
                questions=benchmark.questions[: args.max_questions],
                metadata=dict(benchmark.metadata),
                source_path=benchmark.source_path,
            )
    if not questions:
        raise ValueError("No questions remain after applying --max-questions.")
    return questions, benchmark, str(Path(source).resolve())


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def resolved_config(config: Any) -> dict[str, Any]:
    return _json_ready(asdict(config))


def pipeline_summary(
    args: argparse.Namespace,
    config: Any,
    *,
    question_source: str,
    question_count: int,
) -> dict[str, Any]:
    """Return a compact, generic description of the effective run."""

    return {
        "mode": args.mode or config.phase4_run_mode,
        "question_source": question_source,
        "question_count": question_count,
        "documents": {
            "sample": str(config.sample_data_dir),
            "raw": str(config.raw_data_dir),
            "pdf": str(config.pdf_data_dir),
        },
        "output_directory": str(config.output_root),
        "retrieval": {
            "mode": config.retrieval_mode,
            "dense_top_k": config.dense_top_k,
            "bm25_top_k": config.bm25_top_k,
            "retrieval_top_k": config.retrieval_top_k,
        },
        "vector_database": {
            "directory": str(config.qdrant_dir),
            "collection": config.qdrant_collection_name,
        },
        "embedding": {
            "model": config.embedding_model_name,
            "device": config.embedding_device,
        },
        "reranker": {
            "enabled": config.reranker_enabled,
            "model": config.reranker_model_name,
            "device": config.reranker_device,
            "batch_size": config.reranker_batch_size,
            "local_files_only": config.reranker_local_files_only,
        },
        "llm": {
            "model": config.ollama_model_name,
            "generation_retries": config.generation_retries,
            "retry_cooldown_seconds": config.retry_cooldown_seconds,
            "max_answer_words": config.max_answer_words,
        },
        "checkpoint": {
            "resume": str(args.resume.resolve()) if args.resume else None,
            "enabled": True,
        },
    }


def report_question_source(questions: Sequence[str], source: str) -> None:
    _print(f"Loaded {len(questions)} questions from:")
    _print(source)


def print_dry_run(
    args: argparse.Namespace,
    config: Any,
    questions: Sequence[str],
    source: str,
) -> None:
    """Print validated inputs without constructing any pipeline dependency."""

    _print("Dry run: validation completed; no pipeline components were initialized.")
    _print(f"Output directory:\n{config.output_root}")
    _print("Question preview:")
    for index, question in enumerate(questions[:3], start=1):
        _print(f"  {index}. {question}")
    _print("Pipeline configuration:")
    _print(
        json.dumps(
            pipeline_summary(
                args,
                config,
                question_source=source,
                question_count=len(questions),
            ),
            indent=2,
            sort_keys=True,
        )
    )


def _permission_probe(directory: Path) -> tuple[bool, str]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=True):
            pass
    except OSError as exc:
        return False, str(exc)
    return True, f"Writable: {directory}"


def _health_line(status: str, name: str, message: str) -> None:
    _print(f"{status:<4} {name}: {message}")


def run_health_check(
    args: argparse.Namespace,
    config: Any,
    *,
    questions: Sequence[str] | None,
    source: str | None,
    input_error: Exception | None,
) -> bool:
    """Probe configured resources without executing QA."""

    failures = 0

    configured_source_root = config.project_root / "src"
    if config.project_root.is_dir() and configured_source_root.is_dir():
        _health_line("PASS", "Project structure", str(config.project_root))
    else:
        failures += 1
        _health_line(
            "FAIL",
            "Project structure",
            "Project/source directory is missing; verify --project-root.",
        )

    if input_error is None and questions is not None and source is not None:
        _health_line(
            "PASS",
            "Question source",
            f"{len(questions)} readable questions in {source}",
        )
    else:
        failures += 1
        _health_line(
            "FAIL",
            "Question source",
            f"{input_error} Supply a readable --questions-file or configured benchmark.",
        )

    document_dirs = (
        config.sample_data_dir,
        config.raw_data_dir,
        config.pdf_data_dir,
    )
    document_files = [
        path
        for directory in document_dirs
        if directory.is_dir()
        for pattern in ("*.txt", "*.pdf")
        for path in directory.rglob(pattern)
    ]
    if document_files:
        _health_line(
            "PASS",
            "Document directories",
            f"{len(document_files)} supported files across configured directories.",
        )
    else:
        failures += 1
        _health_line(
            "FAIL",
            "Document directories",
            "No .txt or .pdf documents found; verify configured document directories.",
        )

    vector_parent = (
        config.qdrant_dir
        if config.qdrant_dir.exists()
        else config.qdrant_dir.parent
    )
    if config.qdrant_collection_name.strip() and vector_parent.is_dir():
        _health_line(
            "PASS",
            "Vector database",
            f"Directory {config.qdrant_dir}; collection {config.qdrant_collection_name}",
        )
    else:
        failures += 1
        _health_line(
            "FAIL",
            "Vector database",
            "Configured directory parent or collection name is invalid.",
        )

    if config.reranker_enabled:
        started = perf_counter()
        try:
            from cial_knowledge_os.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                config.reranker_model_name,
                device=config.reranker_device,
                batch_size=config.reranker_batch_size,
                local_files_only=config.reranker_local_files_only,
            )
            reranker.load()
        except Exception as exc:
            failures += 1
            _health_line(
                "FAIL",
                "Reranker",
                f"{exc} Verify the configured model, cache/download policy, and device.",
            )
        else:
            _health_line(
                "PASS",
                "Reranker",
                f"{config.reranker_model_name} via {reranker.load_source} "
                f"on {config.reranker_device} ({perf_counter() - started:.2f}s)",
            )
    else:
        _health_line("WARN", "Reranker", "Disabled by configuration.")

    try:
        from cial_knowledge_os.llm import create_local_llm

        create_local_llm(config)
    except Exception as exc:
        failures += 1
        _health_line(
            "FAIL",
            "LLM",
            f"{exc} Start the configured local service and install the configured model.",
        )
    else:
        _health_line("PASS", "LLM", config.ollama_model_name)

    writable, message = _permission_probe(config.output_root)
    if writable:
        _health_line("PASS", "Output directory", message)
    else:
        failures += 1
        _health_line(
            "FAIL",
            "Output directory",
            f"{message} Choose a writable --output-dir.",
        )

    _print(
        f"Health check complete: {failures} failure(s). "
        "QA execution was not started."
    )
    return failures == 0


def _load_reranker(pipeline: Any, config: Any) -> tuple[str, float]:
    if not config.reranker_enabled:
        return "disabled", 0.0
    started = perf_counter()
    load = getattr(pipeline.reranker, "load", None)
    if not callable(load):
        load = getattr(pipeline.reranker, "_load_model", None)
    if not callable(load):
        return "deferred", perf_counter() - started
    load()
    source = str(getattr(pipeline.reranker, "load_source", None) or "unknown")
    return source, perf_counter() - started


def execute(
    args: argparse.Namespace,
    *,
    config: Any | None = None,
    questions: list[str] | None = None,
    benchmark: Any | None = None,
    source_label: str | None = None,
    reporter: StartupReporter | None = None,
) -> Any:
    """Initialize the existing Phase 4 pipeline and run terminal batch QA."""

    reporter = reporter or StartupReporter(verbose=args.verbose)
    if config is None:
        config = build_config(args)
    if questions is None or source_label is None:
        questions, benchmark, source_label = select_inputs(args, config)
    mode = args.mode or config.phase4_run_mode
    effective_large_run = bool(
        args.large_run
        or (
            mode == "manual_qa"
            and len(questions) > config.max_inline_manual_questions
        )
    )

    reporter.step("Initializing pipeline")
    from cial_knowledge_os.phase4_pipeline import Phase4RAGPipeline
    from cial_knowledge_os.phase4_runner import Phase4Runner

    pipeline = Phase4RAGPipeline(config)
    try:
        reporter.step("Loading documents")
        documents = pipeline.load()

        reporter.step("Building/loading indexes")
        chunks = pipeline.chunk()
        vectors = pipeline.embed()
        pipeline.index()
        reporter.detail(
            json.dumps(
                {
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "vectors": len(vectors),
                },
                sort_keys=True,
            )
        )

        reporter.step("Loading reranker")
        load_source, load_duration = _load_reranker(pipeline, config)
        _print(f"Configured reranker:\n{config.reranker_model_name}")
        _print(f"Load source: {load_source}")
        _print(f"Device: {config.reranker_device}")
        _print(f"Load duration: {load_duration:.2f}s")

        reporter.step("Checking LLM availability")
        _print(f"Configured LLM:\n{config.ollama_model_name}")
        _print("Checking availability...")
        from cial_knowledge_os.llm import create_local_llm

        try:
            pipeline.llm = create_local_llm(config)
        except Exception as exc:
            raise RuntimeError(
                f"Configured LLM '{config.ollama_model_name}' is unavailable. "
                "Start the configured local LLM service, make the configured "
                "model available there, or select another model with "
                "--llm-model/configuration."
            ) from exc
        _print("Availability: available")

        reporter.step("Starting execution")
        result = Phase4Runner(pipeline=pipeline, config=config).run(
            questions=questions,
            benchmark=benchmark,
            run_mode=mode,
            run_metadata={
                "run_label": "terminal_phase4_batch",
                "question_source": source_label,
                "large_run": effective_large_run,
            },
            resume_run=args.resume,
        )
        reporter.detail(
            "Question count returned by Phase4Runner: "
            f"{result.summary.get('question_count', 0)}"
        )
        return result
    finally:
        pipeline.close()


def print_artifact_paths(result: Any) -> None:
    """Print every primary artifact path produced by a completed run."""

    paths = result.paths
    artifacts = (
        ("run", paths.root),
        ("results.csv", paths.results_csv),
        ("results.xlsx", paths.results_xlsx),
        ("report.html", paths.report_html),
        ("config.json", paths.config_json),
        ("summary.json", paths.summary_json),
        ("metrics.json", paths.metrics_json),
        ("retrieval.json", paths.retrieval_json),
        ("logs.txt", paths.logs),
        ("context", paths.context),
        ("figures", paths.figures),
        ("checkpoint.json", paths.root / "checkpoint.json"),
        ("partial_results.csv", paths.root / "partial_results.csv"),
        ("partial_results.jsonl", paths.root / "partial_results.jsonl"),
        ("partial_retrieval.jsonl", paths.root / "partial_retrieval.jsonl"),
    )
    _print("\nPhase 4 batch run complete. Artifacts:")
    for label, path in artifacts:
        _print(f"  {label}: {path}")
    for figure in sorted(paths.figures.iterdir()):
        if figure.is_file() and figure.suffix.casefold() in {".svg", ".html"}:
            _print(f"  visualization: {figure}")
    with paths.results_csv.open(encoding="utf-8-sig", newline="") as handle:
        written_rows = sum(1 for _ in csv.DictReader(handle))
    _print(f"  results.csv question rows: {written_rows}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI or one of its non-executing validation modes."""

    _enable_immediate_output()
    reporter = StartupReporter()
    reporter.step("Phase 4 CLI starting")
    reporter.step("Parsing arguments")
    args = build_parser().parse_args(argv)
    reporter.verbose = args.verbose

    reporter.step("Loading configuration")
    config = build_config(args)
    if args.verbose:
        _print("Resolved configuration:")
        _print(json.dumps(resolved_config(config), indent=2, sort_keys=True))

    reporter.step("Resolving input source")
    questions: list[str] | None = None
    benchmark: Any | None = None
    source_label: str | None = None
    input_error: Exception | None = None
    try:
        questions, benchmark, source_label = select_inputs(args, config)
    except (FileNotFoundError, ValueError) as exc:
        input_error = exc
        if not args.health_check:
            raise
    if questions is not None and source_label is not None:
        report_question_source(questions, source_label)

    reporter.step(f"Output directory: {config.output_root}")

    if args.health_check:
        return (
            0
            if run_health_check(
                args,
                config,
                questions=questions,
                source=source_label,
                input_error=input_error,
            )
            else 1
        )

    assert questions is not None and source_label is not None
    if args.verbose:
        _print("Execution settings:")
        _print(
            json.dumps(
                pipeline_summary(
                    args,
                    config,
                    question_source=source_label,
                    question_count=len(questions),
                ),
                indent=2,
                sort_keys=True,
            )
        )
    if args.dry_run:
        print_dry_run(args, config, questions, source_label)
        reporter.step("Dry run complete")
        return 0

    result = execute(
        args,
        config=config,
        questions=questions,
        benchmark=benchmark,
        source_label=source_label,
        reporter=reporter,
    )
    print_artifact_paths(result)
    reporter.step("Execution complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
