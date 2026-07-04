"""Run Phase 4 batch QA from a terminal and export the full artifact bundle.

This entry point mirrors the Phase 4 notebook's configuration and pipeline
initialization, but deliberately performs no inline trace rendering. Keeping
long-running generation outside Jupyter avoids tying model and vector-store
lifetimes to a notebook kernel while reusing ``Phase4Runner`` as the single
source of truth for exports and visualizations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cial_knowledge_os import (  # noqa: E402
    Benchmark,
    Phase4Config,
    Phase4RAGPipeline,
    Phase4RunResult,
    Phase4Runner,
    load_benchmark,
)


DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "data" / "manual_qa" / "phase4_questions.txt"


def positive_integer(value: str) -> int:
    """Parse a strictly positive CLI integer and return it to argparse.

    The input is one command-line token. The output is an integer greater than
    zero; invalid values become an ``ArgumentTypeError`` with actionable text.
    This keeps batch-size and question-limit validation consistent before any
    local models or corpus data are loaded.
    """

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return parsed


def non_negative_integer(value: str) -> int:
    """Parse a non-negative CLI integer for retry configuration."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def non_negative_number(value: str) -> float:
    """Parse a non-negative CLI number for cooldown configuration."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def load_questions(path: str | Path) -> list[str]:
    """Load non-empty questions from a CSV or line-oriented TXT file.

    ``path`` may point to a UTF-8/UTF-8-BOM CSV containing a ``question``
    column, or to a UTF-8 TXT file with one question per line. The returned list
    preserves file order and strips surrounding whitespace. Empty inputs,
    unsupported extensions, and malformed CSV schemas fail before pipeline
    initialization. This function only prepares Phase 4 runner inputs and does
    not alter notebook or Phase 1--3 behavior.
    """

    source = Path(path).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
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
    """Create and return the terminal interface for Phase 4 batch execution.

    Inputs are parsed from the supported flags; the output is a configured
    ``ArgumentParser``. Defaults match the Phase 4 notebook, while explicit
    flags only override execution mode, question source, run sizing, and
    reranker deployment settings.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 4 outside Jupyter and export the complete batch QA bundle."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "manual_qa", "benchmark"),
        default="manual_qa",
        help="Execution mode recorded in the run artifacts (default: manual_qa).",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="Optional .csv with a question column or .txt with one question per line.",
    )
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
        help="Run at most the first N questions after loading the selected source.",
    )
    parser.add_argument(
        "--reranker-device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Cross-encoder execution device (default: auto).",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=positive_integer,
        default=16,
        help="Cross-encoder scoring batch size (default: 16).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable reranker downloads and require an existing Hugging Face cache.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted Phase 4 run from its run folder.",
    )
    parser.add_argument(
        "--generation-retries",
        type=non_negative_integer,
        default=2,
        help="Generation retries after the initial attempt (default: 2).",
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=non_negative_number,
        default=20.0,
        help="Cooldown between retryable generation attempts (default: 20).",
    )
    parser.add_argument(
        "--max-answer-words",
        type=positive_integer,
        help="Optional upper word target for each generated answer.",
    )
    return parser


def build_config(args: argparse.Namespace) -> Phase4Config:
    """Build the effective Phase 4 configuration from parsed CLI arguments.

    ``args`` must come from :func:`build_parser`. The returned ``Phase4Config``
    mirrors the notebook's hybrid retrieval, reranking, selection, token, and
    trace settings. Only the documented CLI overrides differ. Its standard
    Phase 4 output root preserves all existing export paths and remains
    backward compatible with ``Phase4Runner``.
    """

    return Phase4Config(
        project_root=PROJECT_ROOT,
        retrieval_mode="hybrid",
        dense_top_k=10,
        bm25_top_k=10,
        retrieval_top_k=10,
        rrf_k=60,
        max_context_tokens=4096,
        reranker_candidate_top_k=30,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_local_files_only=args.local_files_only,
        min_selected_evidence=3,
        max_selected_evidence=8,
        reranker_score_threshold=-4.0,
        fallback_to_top_n_if_empty=True,
        fallback_top_n=3,
        weak_evidence_answer_allowed=True,
        answer_detail_level="detailed",
        min_answer_words=250,
        max_answer_words=args.max_answer_words,
        prefer_structured_answers=True,
        include_decision_notes=True,
        generation_retries=args.generation_retries,
        retry_cooldown_seconds=args.retry_cooldown_seconds,
        evidence_token_budget=2400,
        selected_evidence_target_min_tokens=800,
        selected_evidence_target_max_tokens=1500,
        evidence_max_chunks_per_source=2,
        evidence_redundancy_threshold=0.85,
        phase4_trace_mode="full",
        phase4_run_mode=args.mode,
        # The 25-question guard protects interactive notebook rendering. This
        # process is the non-interactive batch surface, so every loaded question
        # proceeds unless --max-questions explicitly sliced the input above.
        allow_large_run=True,
    )


def select_inputs(
    args: argparse.Namespace,
    config: Phase4Config,
) -> tuple[list[str] | None, Benchmark | None, str]:
    """Resolve CLI question inputs and optional benchmark expectations.

    Inputs are parsed arguments and the effective config. Outputs are a
    question list, an optional ``Benchmark``, and a human-readable source
    description. CSV benchmark mode preserves expected answers and metadata;
    TXT benchmark mode can execute and export but cannot calculate qualification
    metrics because the format carries questions only.
    """

    benchmark: Benchmark | None = None
    source_label = str(DEFAULT_QUESTIONS_FILE)

    if args.questions_file is not None:
        source = args.questions_file.expanduser()
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        source = source.resolve()
        source_label = str(source)
        if args.mode == "benchmark" and source.suffix.casefold() == ".csv":
            benchmark = load_benchmark(source)
            questions = [item.question for item in benchmark.questions]
        else:
            questions = load_questions(source)
    elif args.mode == "benchmark":
        metadata_path = (
            config.benchmark_metadata_path
            if config.benchmark_metadata_path.is_file()
            else None
        )
        benchmark = load_benchmark(
            config.benchmark_csv_path,
            metadata_path=metadata_path,
        )
        questions = [item.question for item in benchmark.questions]
        source_label = str(config.benchmark_csv_path)
    else:
        try:
            # Manual inputs are data, not application code. Keeping the default
            # list in a text file makes QA changes reviewable and reusable
            # without editing the terminal runner.
            questions = load_questions(DEFAULT_QUESTIONS_FILE)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                "The default Phase 4 questions file is missing or empty. "
                f"Expected path: {DEFAULT_QUESTIONS_FILE}. "
                "Create a UTF-8 text file with one question per line, or pass "
                "--questions-file <path> to use a different CSV/TXT file."
            ) from exc

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
    return questions, benchmark, source_label


def execute(args: argparse.Namespace) -> Phase4RunResult:
    """Initialize the notebook-equivalent pipeline and run terminal batch QA.

    Parsed CLI arguments are the input. The returned ``Phase4RunResult`` points
    to the complete CSV/XLSX/HTML/JSON/log/context/figure bundle. Corpus loading,
    chunking, embedding, and indexing intentionally match the notebook order;
    no trace is rendered inline, and prior notebook/API behavior is unchanged.
    """

    config = build_config(args)
    questions, benchmark, source_label = select_inputs(args, config)
    effective_large_run = bool(
        args.large_run
        or (
            args.mode == "manual_qa"
            and len(questions) > config.max_inline_manual_questions
        )
    )

    print("Initializing Phase 4 local pipeline...")
    print(f"Loaded question count: {len(questions)}")
    print(f"Question count entering Phase4Runner: {len(questions)}")
    print(
        {
            "mode": args.mode,
            "question_count": len(questions),
            "question_source": source_label,
            "large_run": effective_large_run,
            "reranker_device": config.reranker_device,
            "reranker_batch_size": config.reranker_batch_size,
            "local_files_only": config.reranker_local_files_only,
            "generation_retries": config.generation_retries,
            "generation_attempts": config.generation_retries + 1,
            "retry_cooldown_seconds": config.retry_cooldown_seconds,
            "max_answer_words": config.max_answer_words,
            "resume": str(args.resume.resolve()) if args.resume else None,
        }
    )

    pipeline = Phase4RAGPipeline(config)
    try:
        documents = pipeline.load()
        chunks = pipeline.chunk()
        vectors = pipeline.embed()
        pipeline.index()
        print(
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "vectors": len(vectors),
            }
        )

        result = Phase4Runner(pipeline=pipeline, config=config).run(
            questions=questions,
            benchmark=benchmark,
            run_mode=args.mode,
            run_metadata={
                "run_label": "terminal_phase4_batch",
                "question_source": source_label,
                "large_run": effective_large_run,
            },
            resume_run=args.resume,
        )
        print(
            "Question count returned by Phase4Runner: "
            f"{result.summary.get('question_count', 0)}"
        )
        return result
    finally:
        # A terminal process should release embedded Qdrant deterministically;
        # relying on interpreter shutdown can leave a noisy destructor warning
        # or delay the next process from acquiring the local storage lock.
        pipeline.close()


def print_artifact_paths(result: Phase4RunResult) -> None:
    """Print every primary artifact path produced by a completed Phase 4 run.

    The input is a completed run result. Output is terminal-only status text;
    files are not modified. Figure SVGs are enumerated from the existing runner
    directory so newly supported visualizations appear automatically.
    """

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
    print("\nPhase 4 batch run complete. Artifacts:")
    for label, path in artifacts:
        print(f"  {label}: {path}")
    for figure in sorted(paths.figures.iterdir()):
        if figure.is_file() and figure.suffix.casefold() in {".svg", ".html"}:
            print(f"  visualization: {figure}")
    with paths.results_csv.open(encoding="utf-8-sig", newline="") as handle:
        written_rows = sum(1 for _ in csv.DictReader(handle))
    print(f"  results.csv question rows: {written_rows}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse optional CLI arguments, execute Phase 4, and return a status code.

    ``argv`` supports direct automated testing; ``None`` reads the real command
    line. Successful execution returns ``0`` after printing artifact paths.
    Runtime errors are intentionally allowed to surface with their actionable
    model, corpus, Ollama, or file diagnostics.
    """

    args = build_parser().parse_args(argv)
    result = execute(args)
    print_artifact_paths(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
