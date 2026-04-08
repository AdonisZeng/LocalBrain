"""
Base module for vector store providers.
Re-exports the abstract interface.
"""

from app.core.interfaces import VectorStoreProvider, VectorStoreInterface

__all__ = ["VectorStoreProvider", "VectorStoreInterface"]
