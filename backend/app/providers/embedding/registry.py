"""
Embedding Provider Registry.
Inherits shared logic from BaseProviderRegistry.
"""

from typing import Dict, Type, Optional, Any, List, Callable
from app.core.interfaces import EmbeddingProvider
from app.core.logging_config import get_logger
from app.providers.base import BaseProviderRegistry

logger = get_logger("embedding_registry")


class EmbeddingProviderRegistry(BaseProviderRegistry):
    """Registry for embedding providers."""

    _provider_classes: Dict[str, Type[EmbeddingProvider]] = {}
    _provider_instances: Dict[str, EmbeddingProvider] = {}

    @classmethod
    def get_provider_interface(cls) -> Type:
        return EmbeddingProvider

    @classmethod
    def create_embeddings(cls, name: str, config: Dict[str, Any]):
        """Create an embeddings instance using the specified provider."""
        provider = cls.get_provider(name)
        if provider is None:
            raise ValueError(f"Unknown embedding provider: {name}. Available: {cls.list_providers()}")
        return provider.create_embeddings(config)

    @classmethod
    def get_provider_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider."""
        provider = cls.get_provider(name)
        if provider is None:
            return None
        return {
            "name": provider.name,
            "dimension": provider.dimension,
        }


def register_embedding_provider(name: str) -> Callable:
    """Decorator to register an embedding provider."""
    def decorator(cls: Type[EmbeddingProvider]) -> Type[EmbeddingProvider]:
        EmbeddingProviderRegistry.register(name, cls)
        return cls
    return decorator
