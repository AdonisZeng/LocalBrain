"""
LLM Provider Registry.
Inherits shared logic from BaseProviderRegistry.
"""

from typing import Dict, Type, Optional, Any, List, Callable
from app.core.interfaces import LLMProvider
from app.core.logging_config import get_logger
from app.providers.base import BaseProviderRegistry

logger = get_logger("llm_registry")


class LLMProviderRegistry(BaseProviderRegistry):
    """Registry for LLM providers."""

    _provider_classes: Dict[str, Type[LLMProvider]] = {}
    _provider_instances: Dict[str, LLMProvider] = {}

    @classmethod
    def get_provider_interface(cls) -> Type:
        return LLMProvider

    @classmethod
    def create_llm(cls, name: str, config: Dict[str, Any]):
        """Create an LLM instance using the specified provider."""
        provider = cls.get_provider(name)
        if provider is None:
            raise ValueError(f"Unknown LLM provider: {name}. Available: {cls.list_providers()}")
        return provider.create_llm(config)

    @classmethod
    def get_provider_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider."""
        provider = cls.get_provider(name)
        if provider is None:
            return None
        return {
            "name": provider.name,
            "supported_models": provider.supported_models,
        }


def register_llm_provider(name: str) -> Callable:
    """Decorator to register an LLM provider."""
    def decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        LLMProviderRegistry.register(name, cls)
        return cls
    return decorator
