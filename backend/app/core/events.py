"""
Event bus for hot reload functionality.
Implements the Observer pattern for configuration change notifications.
"""

from typing import Callable, Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from app.core.logging_config import get_logger

logger = get_logger("event_bus")

# Shared thread pool for async event emission without blocking
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eventbus_")
    return _executor


class ConfigEvent(Enum):
    """Configuration change events."""
    LLM_CONFIG_CHANGED = "llm_config_changed"
    EMBEDDING_CONFIG_CHANGED = "embedding_config_changed"
    VECTORSTORE_CONFIG_CHANGED = "vectorstore_config_changed"
    DOCUMENT_PROCESSING_CHANGED = "document_processing_changed"
    FULL_RELOAD = "full_reload"


@dataclass
class Event:
    """Event payload."""
    type: ConfigEvent
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Central event bus for hot reload functionality.

    Usage:
        # Subscribe to events
        def on_llm_change(event):
            llm_service.reload_config()

        EventBus.subscribe(ConfigEvent.LLM_CONFIG_CHANGED, on_llm_change)

        # Emit events
        EventBus.emit(ConfigEvent.LLM_CONFIG_CHANGED, {"provider": "ollama"})
    """

    _subscribers: Dict[ConfigEvent, List[Callable]] = {}
    _lock = threading.Lock()

    @classmethod
    def subscribe(cls, event_type: ConfigEvent, callback: Callable) -> None:
        """Subscribe to an event type."""
        with cls._lock:
            if event_type not in cls._subscribers:
                cls._subscribers[event_type] = []

            if callback not in cls._subscribers[event_type]:
                cls._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed to {event_type.value}: {callback.__name__}")

    @classmethod
    def unsubscribe(cls, event_type: ConfigEvent, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        with cls._lock:
            if event_type in cls._subscribers:
                try:
                    cls._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type.value}: {callback.__name__}")
                except ValueError:
                    pass

    @classmethod
    def emit(cls, event_type: ConfigEvent, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to all subscribers (synchronous)."""
        event = Event(type=event_type, timestamp=datetime.now(), data=data or {})

        with cls._lock:
            subscribers = cls._subscribers.get(event_type, []).copy()

        if not subscribers:
            logger.debug(f"No subscribers for event: {event_type.value}")
            return

        logger.info(f"Emitting event: {event_type.value} to {len(subscribers)} subscriber(s)")

        for callback in subscribers:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    logger.warning(f"Async callback {callback.__name__} used with sync emit")
                logger.debug(f"Event {event_type.value} processed by {callback.__name__}")
            except Exception as e:
                logger.error(f"Error in event callback {callback.__name__}: {e}", exc_info=True)

    @classmethod
    def emit_async(cls, event_type: ConfigEvent, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event asynchronously (non-blocking)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(cls._emit_event_async(event_type, data))
            else:
                loop.run_until_complete(cls._emit_event_async(event_type, data))
        except RuntimeError:
            # No event loop available, use shared thread pool
            executor = _get_executor()
            executor.submit(cls.emit, event_type, data)

    @classmethod
    async def _emit_event_async(cls, event_type: ConfigEvent, data: Optional[Dict[str, Any]] = None) -> None:
        """Internal async event emission."""
        event = Event(type=event_type, timestamp=datetime.now(), data=data or {})

        with cls._lock:
            subscribers = cls._subscribers.get(event_type, []).copy()

        if not subscribers:
            logger.debug(f"No subscribers for async event: {event_type.value}")
            return

        logger.info(f"Emitting async event: {event_type.value} to {len(subscribers)} subscriber(s)")

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                logger.debug(f"Async event {event_type.value} processed by {callback.__name__}")
            except Exception as e:
                logger.error(f"Error in async event callback {callback.__name__}: {e}", exc_info=True)

    @classmethod
    def clear_subscribers(cls) -> None:
        """Clear all subscribers. Mainly for testing."""
        with cls._lock:
            cls._subscribers.clear()

    @classmethod
    def list_subscribers(cls) -> Dict[str, int]:
        """List all subscribers by event type. For debugging."""
        with cls._lock:
            return {et.value: len(subs) for et, subs in cls._subscribers.items()}


class EventBusObserver:
    """
    Mixin for services that auto-subscribe to config change events.

    Usage:
        class MyService(EventBusObserver):
            _subscribed_event_type = ConfigEvent.LLM_CONFIG_CHANGED

            def __init__(self):
                super().__init__()
                # Subscription happens automatically

            def on_config_changed(self, event: Event):
                self.reload_config()
    """

    _subscribed_event_type: Optional[ConfigEvent] = None

    def on_config_changed(self, event: Event) -> None:
        """Handle config change event. Override in subclass."""
        raise NotImplementedError

    def _subscribe_to_events(self) -> None:
        """Subscribe to configured events. Call from subclass __init__."""
        if self._subscribed_event_type is not None:
            EventBus.subscribe(self._subscribed_event_type, self.on_config_changed)
