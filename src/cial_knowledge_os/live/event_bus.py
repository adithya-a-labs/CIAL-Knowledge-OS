"""Thread-safe in-process publish/subscribe event bus."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from threading import RLock
from typing import Any
from uuid import uuid4

from .schemas import LiveEvent

EventCallback = Callable[[dict[str, Any]], None]


class EventBus:
    """Publish synchronous pipeline events and stream them asynchronously."""

    def __init__(self, *, history_size: int = 1_000) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be greater than zero.")
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: dict[str, EventCallback] = {}
        self._sequence = 0
        self._lock = RLock()

    def publish(self, event: LiveEvent | Mapping[str, Any]) -> dict[str, Any]:
        value = (
            event.to_dict()
            if isinstance(event, LiveEvent)
            else LiveEvent.from_mapping(event).to_dict()
        )
        with self._lock:
            self._sequence += 1
            value["sequence"] = self._sequence
            self._history.append(value)
            subscribers = list(self._subscribers.values())
        for callback in subscribers:
            try:
                callback(dict(value))
            except Exception:
                # Observers cannot interrupt the Phase 5 execution path.
                continue
        return dict(value)

    def subscribe(self, callback: EventCallback) -> str:
        token = uuid4().hex
        with self._lock:
            self._subscribers[token] = callback
        return token

    def unsubscribe(self, token: str) -> None:
        with self._lock:
            self._subscribers.pop(token, None)

    def history(self, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in self._history
                if int(item.get("sequence") or 0) > after_sequence
            ]

    async def stream(
        self, *, after_sequence: int = 0
    ) -> AsyncIterator[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def enqueue(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        token = self.subscribe(enqueue)
        try:
            last_sequence = after_sequence
            for event in self.history(after_sequence=after_sequence):
                last_sequence = max(
                    last_sequence, int(event.get("sequence") or 0)
                )
                yield event
            while True:
                event = await queue.get()
                sequence = int(event.get("sequence") or 0)
                if sequence <= last_sequence:
                    continue
                last_sequence = sequence
                yield event
        finally:
            self.unsubscribe(token)
