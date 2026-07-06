"""Failure-isolated, in-process publish/subscribe support."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from threading import RLock

from .events import ExecutionEvent

EventHandler = Callable[[ExecutionEvent], None]


class EventBus:
    """Deliver events synchronously without allowing observers to affect work."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._handlers: list[EventHandler] = []
        self._lock = RLock()
        self.handler_warnings: list[str] = []

    def subscribe(self, handler: EventHandler) -> EventHandler:
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
        return handler

    def unsubscribe(self, handler: EventHandler) -> None:
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def emit(self, event: ExecutionEvent) -> ExecutionEvent:
        if not self.enabled:
            return event
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                message = (
                    f"EOF handler {getattr(handler, '__name__', type(handler).__name__)} "
                    f"failed: {type(exc).__name__}: {exc}"
                )
                self.handler_warnings.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        return event
