"""Optional Rich and dependency-free terminal progress renderers."""

from __future__ import annotations

import sys
from time import monotonic
from typing import Any, TextIO

from .events import ExecutionEvent
from .progress import ProgressTracker


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    value = max(0, int(seconds))
    return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"


class PlainConsoleRenderer:
    def __init__(
        self,
        tracker: ProgressTracker,
        *,
        stream: TextIO | None = None,
        refresh_seconds: float = 1.0,
    ) -> None:
        self.tracker = tracker
        self.stream = stream or sys.stdout
        self.refresh_seconds = refresh_seconds
        self._last_rendered = 0.0

    def __call__(self, event: ExecutionEvent) -> None:
        now = monotonic()
        important = event.event_type in {
            "run_started",
            "run_resumed",
            "run_completed",
            "run_failed",
            "question_completed",
            "question_failed",
            "warning",
            "error",
        }
        if not important and now - self._last_rendered < self.refresh_seconds:
            return
        self._last_rendered = now
        value = self.tracker.snapshot()
        if event.event_type in {"run_started", "run_resumed"}:
            self.stream.write(
                f"CIAL Knowledge OS — {value['phase'] or 'Execution'}\n"
                f"Run: {value['run_id']}\n"
            )
        self.stream.write(
            "[EOF] "
            f"{value['completed']}/{value['total']} "
            f"({value['percent']:.1f}%) "
            f"stage={value['current_stage'] or '-'} "
            f"elapsed={_duration(value['elapsed'])} "
            f"eta={_duration(value['eta'])} "
            f"warnings={value['warnings']} errors={value['errors']}\n"
        )
        self.stream.flush()


class RichConsoleRenderer(PlainConsoleRenderer):
    """Compact Rich output; lifecycle remains identical to plain rendering."""

    def __init__(self, tracker: ProgressTracker, **kwargs: Any) -> None:
        super().__init__(tracker, **kwargs)
        from rich.console import Console

        self.console = Console(file=self.stream)

    def __call__(self, event: ExecutionEvent) -> None:
        now = monotonic()
        if (
            event.event_type not in {"run_started", "run_completed", "run_failed",
                                     "question_completed", "question_failed"}
            and now - self._last_rendered < self.refresh_seconds
        ):
            return
        self._last_rendered = now
        value = self.tracker.snapshot()
        if event.event_type in {"run_started", "run_resumed"}:
            self.console.print(
                f"[bold]CIAL Knowledge OS — "
                f"{value['phase'] or 'Execution'}[/bold]\n"
                f"Run: {value['run_id']}"
            )
        self.console.print(
            f"[cyan]EOF[/cyan] {value['completed']}/{value['total']} "
            f"[{value['percent']:.1f}%]  "
            f"{value['current_stage'] or '-'}  "
            f"elapsed {_duration(value['elapsed'])}  "
            f"ETA {_duration(value['eta'])}"
        )


def create_console_renderer(
    tracker: ProgressTracker,
    *,
    rich: str | bool = "auto",
    stream: TextIO | None = None,
    refresh_seconds: float = 1.0,
) -> PlainConsoleRenderer:
    if rich is not False:
        try:
            import rich as _rich  # noqa: F401
            return RichConsoleRenderer(
                tracker,
                stream=stream,
                refresh_seconds=refresh_seconds,
            )
        except ImportError:
            pass
    return PlainConsoleRenderer(
        tracker,
        stream=stream,
        refresh_seconds=refresh_seconds,
    )
