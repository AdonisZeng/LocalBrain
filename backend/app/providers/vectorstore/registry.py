"""
VectorStore Provider Registry.
Inherits shared logic from BaseProviderRegistry.
"""

from typing import Dict, Type, Optional, Any, List, Callable
from app.core.interfaces import VectorStoreProvider, VectorStoreInterface
from app.core.logging_config import get_logger
from app.providers.base import BaseProviderRegistry

logger = get_logger("vectorstore_registry")


class VectorStoreProviderRegistry(BaseProviderRegistry):
    """Registry for vector store providers."""

    _provider_classes: Dict[str, Type[VectorStoreProvider]] = {}
    _provider_instances: Dict[str, VectorStoreProvider] = {}

    @classmethod
    def get_provider_interface(cls) -> Type:
        return VectorStoreProvider

    @classmethod
    def create_vectorstore(
        cls,
        name: str,
        embedding_function,
        config: Dict[str, Any]
    ) -> VectorStoreInterface:
        """Create a vector store instance using the specified provider."""
        provider = cls.get_provider(name)
        if provider is None:
            raise ValueError(f"Unknown vector store provider: {name}. Available: {cls.list_providers()}")
        return provider.create_vectorstore(embedding_function, config)

    @classmethod
    def get_provider_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider."""
        provider = cls.get_provider(name)
        if provider is None:
            return None
        return {
            "name": provider.name,
        }


def register_vectorstore_provider(name: str) -> Callable:
    """Decorator to register a vector store provider."""
    def decorator(cls: Type[VectorStoreProvider]) -> Type[VectorStoreProvider]:
        VectorStoreProviderRegistry.register(name, cls)
        return cls
    return decorator
