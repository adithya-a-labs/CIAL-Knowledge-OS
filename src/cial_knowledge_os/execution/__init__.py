"""Execution & Observability Framework (EOF).

EOF observes pipeline work through structured events. It never schedules,
retries, branches, or otherwise decides that work.
"""

from .event_bus import EventBus
from .events import EVENT_TYPES, ExecutionEvent
from .manager import ExecutionManager, NoOpExecutionManager
from .metrics import MetricsCollector
from .progress import ProgressTracker
from .schemas import ExecutionOptions
from .telemetry import TelemetryCollector

__all__ = [
    "EVENT_TYPES",
    "EventBus",
    "ExecutionEvent",
    "ExecutionManager",
    "ExecutionOptions",
    "MetricsCollector",
    "NoOpExecutionManager",
    "ProgressTracker",
    "TelemetryCollector",
]
