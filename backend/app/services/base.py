"""
Base class for model services with hot-reload support.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.events import Event, ConfigEvent, EventBus

logger = None


def _get_logger():
    global logger
    if logger is None:
        from app.core.logging_config import get_logger
        logger = get_logger("base_model_service")
    return logger


class BaseModelService(ABC):
    """
    Base class for model services with hot-reload support.

    Provides common patterns for:
    - Configuration loading and caching
    - Event bus subscription for hot reload
    - Instance caching
    """

    _subscribed_event_type: Optional["ConfigEvent"] = None
    _default_provider: str = "openai"

    def __init__(self):
        self._instance: Optional[Any] = None
        self._config: Optional[dict] = None
        self._provider: str = self._default_provider
        self._load_config()
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to configured events. Called from __init__."""
        from app.core.events import EventBus
        if self._subscribed_event_type is not None:
            EventBus.subscribe(self._subscribed_event_type, self.on_config_changed)

    @abstractmethod
    def _load_config(self) -> None:
        """Load configuration from config file. Override in subclass."""
        pass

    @abstractmethod
    def _create_instance(self) -> Any:
        """Create the underlying service instance. Override in subclass."""
        pass

    def on_config_changed(self, event: "Event") -> None:
        """Handle config change event. Calls reload_config."""
        _get_logger().info(f"Config change event received: {event.data}")
        self.reload_config()

    def reload_config(self) -> None:
        """Reload configuration and clear cached instance."""
        _get_logger().info("Reloading config")
        self._load_config()
        self._instance = None
        _get_logger().info("Instance cleared, will be recreated on next use")

    def get_instance(self) -> Any:
        """Get cached instance, creating if necessary."""
        if self._instance is None:
            _get_logger().debug("Creating new instance")
            self._instance = self._create_instance()
        return self._instance
