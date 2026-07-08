"""Run a complete Phase 4 batch from user-edited defaults or CLI overrides."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


# =====================================================
# USER CONFIGURATION
# Edit these values for day-to-day Phase 4 runs.
# =====================================================

QUESTIONS_FILE = (
    PROJECT_ROOT / "data" / "manual_qa" / "CIAL_Enterprise_Long_Horizon_200_Questions.txt"
)
RUN_MODE = "manual_qa"
MAX_ANSWER_WORDS = 1200
ADAPTIVE_ANSWER_SECTIONS = True
GENERATION_RETRIES = 2
RETRY_COOLDOWN_SECONDS = 20
RERANKER_DEVICE = "auto"
RERANKER_BATCH_SIZE = 16
LOCAL_FILES_ONLY = False
FORCE_REBUILD_INDEX = False
RESUME_RUN_FOLDER = None
QDRANT_MODE = "server"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None
QDRANT_BATCH_SIZE = 32  # Recommended server value; embedded defaults to 256.
QDRANT_UPSERT_WAIT = True


def _print(message: str = "") -> None:
    """Print progress immediately in terminals and VS Code."""

    print(message, flush=True)


def _enable_immediate_output() -> None:
    """Use line-buffered, write-through output where Python supports it."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except (OSError, ValueError):
                pass


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
    """Resolve paths consistently regardless of the launch working directory."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def load_questions(
    path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    """Load a TXT question list or a CSV containing a ``question`` column."""

    source = resolve_path(path, project_root=project_root)
    guidance = (
        f"Expected path: {source}\n"
        "Use a TXT file with one question per line, or a CSV file with a "
        "'question' column. Edit QUESTIONS_FILE in the USER CONFIGURATION "
        "section or pass --questions-file."
    )
    if not source.is_file():
        raise FileNotFoundError(f"Questions file is missing.\n{guidance}")

    if source.suffix.casefold() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "question" not in reader.fieldnames:
                raise ValueError(
                    "Questions CSV must contain a 'question' column.\n"
                    f"{guidance}"
                )
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
        raise ValueError(
            "Questions file must use a .txt or .csv extension.\n"
            f"{guidance}"
        )

    if not questions:
        raise ValueError(f"Questions file is empty.\n{guidance}")
    return questions


def build_parser() -> argparse.ArgumentParser:
    """Provide optional one-off overrides for the user configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 4 using the USER CONFIGURATION defaults in this file. "
            "Every argument is optional."
        )
    )
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument(
        "--mode",
        choices=("smoke", "manual_qa", "benchmark"),
    )
    parser.add_argument("--max-questions", type=positive_integer)
    parser.add_argument("--max-answer-words", type=positive_integer)
    parser.add_argument(
        "--generation-retries",
        type=non_negative_integer,
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=non_negative_number,
    )
    parser.add_argument(
        "--reranker-device",
        choices=("cpu", "cuda", "auto"),
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=positive_integer,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--force-rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore the document manifest and safely rebuild the vector index.",
    )
    parser.add_argument(
        "--large-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _value(override: Any, configured: Any) -> Any:
    return configured if override is None else override


def build_config(args: argparse.Namespace) -> Any:
    """Build Phase4Config primarily from the USER CONFIGURATION constants."""

    from cial_knowledge_os.config import Phase4Config

    return Phase4Config(
        project_root=PROJECT_ROOT,
        phase4_run_mode=_value(args.mode, RUN_MODE),
        max_answer_words=_value(args.max_answer_words, MAX_ANSWER_WORDS),
        adaptive_answer_sections=ADAPTIVE_ANSWER_SECTIONS,
        generation_retries=_value(
            args.generation_retries,
            GENERATION_RETRIES,
        ),
        retry_cooldown_seconds=_value(
            args.retry_cooldown_seconds,
            RETRY_COOLDOWN_SECONDS,
        ),
        reranker_device=_value(args.reranker_device, RERANKER_DEVICE),
        reranker_batch_size=_value(
            args.reranker_batch_size,
            RERANKER_BATCH_SIZE,
        ),
        reranker_local_files_only=_value(
            args.local_files_only,
            LOCAL_FILES_ONLY,
        ),
        force_rebuild_index=_value(
            args.force_rebuild_index,
            FORCE_REBUILD_INDEX,
        ),
        qdrant_mode=QDRANT_MODE,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_batch_size=QDRANT_BATCH_SIZE,
        qdrant_upsert_wait=QDRANT_UPSERT_WAIT,
        # The notebook guard protects interactive rendering. This script is the
        # intentionally unbounded batch surface.
        allow_large_run=True,
    )


def select_inputs(
    args: argparse.Namespace,
    config: Any,
) -> tuple[list[str], Any | None, str]:
    """Load the configured questions and preserve benchmark CSV expectations."""

    source = resolve_path(
        _value(args.questions_file, QUESTIONS_FILE),
        project_root=config.project_root,
    )
    mode = config.phase4_run_mode
    benchmark = None
    questions = load_questions(source, project_root=config.project_root)

    if mode == "benchmark" and source.suffix.casefold() == ".csv":
        from cial_knowledge_os.benchmark_loader import load_benchmark

        benchmark = load_benchmark(source)
        questions = [item.question for item in benchmark.questions]

    if args.max_questions is not None:
        questions = questions[: args.max_questions]
        if benchmark is not None:
            from cial_knowledge_os.benchmark_loader import Benchmark

            benchmark = Benchmark(
                questions=benchmark.questions[: args.max_questions],
                metadata=dict(benchmark.metadata),
                source_path=benchmark.source_path,
            )
    if not questions:
        raise ValueError(
            f"No questions remain after applying --max-questions.\n"
            f"Expected path: {source}\n"
            "Use a TXT file with one question per line, or a CSV file with a "
            "'question' column."
        )
    return questions, benchmark, str(source)


def report_question_source(questions: Sequence[str], source: str) -> None:
    _print(f"Loaded {len(questions)} questions from {source}")


def _resume_path(args: argparse.Namespace) -> Path | None:
    value = _value(args.resume, RESUME_RUN_FOLDER)
    return resolve_path(value) if value is not None else None


def execute(
    args: argparse.Namespace,
    *,
    config: Any,
    questions: list[str],
    benchmark: Any | None,
    source_label: str,
) -> Any:
    """Initialize the existing pipeline and execute the existing runner."""

    _print("Initializing pipeline")
    from cial_knowledge_os.phase4_pipeline import Phase4RAGPipeline
    from cial_knowledge_os.phase4_runner import Phase4Runner
    from cial_knowledge_os.execution import ExecutionManager

    execution_manager = ExecutionManager.from_config(
        config,
        phase="Phase 4",
        run_mode=config.phase4_run_mode,
    )
    manual_question_limit = config.max_inline_manual_questions
    pipeline = Phase4RAGPipeline(config)
    pipeline.execution_manager = execution_manager
    try:
        pipeline.load()
        pipeline.chunk()
        pipeline.embed()
        pipeline.index()
        _print("Indexing complete")

        _print("Starting QA")
        result = Phase4Runner(
            pipeline=pipeline,
            config=config,
            execution_manager=execution_manager,
        ).run(
            questions=questions,
            benchmark=benchmark,
            run_mode=config.phase4_run_mode,
            run_metadata={
                "run_label": "terminal_phase4_batch",
                "question_source": source_label,
                "large_run": (
                    config.phase4_run_mode == "manual_qa"
                    and manual_question_limit is not None
                    and len(questions) > manual_question_limit
                ),
            },
            resume_run=_resume_path(args),
        )
        return result
    finally:
        pipeline.close()


def print_artifact_paths(result: Any) -> None:
    """Print the complete Phase 4 artifact bundle."""

    paths = result.paths
    artifacts = (
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
    for label, path in artifacts:
        _print(f"  {label}: {path}")
    for figure in sorted(paths.figures.iterdir()):
        if figure.is_file() and figure.suffix.casefold() in {".svg", ".html"}:
            _print(f"  visualization: {figure}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 4 directly; CLI arguments are optional overrides only."""

    _enable_immediate_output()
    _print("Starting Phase 4 batch run")
    args = build_parser().parse_args(argv)
    config = build_config(args)
    questions, benchmark, source_label = select_inputs(args, config)
    report_question_source(questions, source_label)

    result = execute(
        args,
        config=config,
        questions=questions,
        benchmark=benchmark,
        source_label=source_label,
    )
    _print(f"Exported run: {result.paths.root}")
    print_artifact_paths(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
