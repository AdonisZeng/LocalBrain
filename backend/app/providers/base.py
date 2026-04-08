"""
Base classes for provider registries.
Provides shared registry functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any, List, TYPE_CHECKING
from app.core.logging_config import get_logger

if TYPE_CHECKING:
    from app.core.interfaces import LLMProvider, EmbeddingProvider, VectorStoreProvider

logger = get_logger("provider_registry_base")


class BaseProviderRegistry(ABC):
    """
    Base class for provider registries.

    Provides shared registration, lookup, and instance caching logic.
    Each subclass must implement the abstract methods for type-specific behavior.
    """

    _provider_classes: Dict[str, Type] = {}
    _provider_instances: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type) -> None:
        """
        Register a new provider class.

        Args:
            name: Provider identifier
            provider_class: Class implementing the provider interface

        Raises:
            TypeError: If provider_class doesn't inherit from the required interface
        """
        interface = cls.get_provider_interface()
        if not issubclass(provider_class, interface):
            raise TypeError(f"{provider_class.__name__} must inherit from {interface.__name__}")

        cls._provider_classes[name] = provider_class
        logger.info(f"Registered {cls.__name__} provider: {name} ({provider_class.__name__})")

    @classmethod
    def get_provider(cls, name: str) -> Optional[Any]:
        """
        Get provider instance (cached per process lifetime).

        Args:
            name: Provider identifier

        Returns:
            Provider instance, or None if not registered
        """
        if name not in cls._provider_instances:
            provider_class = cls._provider_classes.get(name)
            if provider_class:
                cls._provider_instances[name] = provider_class()
                logger.debug(f"Created provider instance: {name}")
            else:
                logger.warning(f"Provider not found: {name}")
                return None
        return cls._provider_instances.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._provider_classes.keys())

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached provider instances. For testing."""
        cls._provider_instances.clear()
        logger.debug(f"Cleared {cls.__name__} instances")

    @classmethod
    @abstractmethod
    def get_provider_interface(cls) -> Type:
        """Return the provider interface class for type checking."""
        pass

    @classmethod
    @abstractmethod
    def get_provider_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider. Each subclass can override."""
        pass
