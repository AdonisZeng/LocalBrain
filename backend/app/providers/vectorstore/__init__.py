# VectorStore Providers package
# Import all providers to trigger @register_vectorstore_provider decorators
from app.providers.vectorstore import chroma

from app.providers.vectorstore.registry import VectorStoreProviderRegistry, register_vectorstore_provider

__all__ = ["VectorStoreProviderRegistry", "register_vectorstore_provider"]
