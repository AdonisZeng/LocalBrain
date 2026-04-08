"""
Base module for embedding providers.
Re-exports the abstract interface.
"""

from app.core.interfaces import EmbeddingProvider

__all__ = ["EmbeddingProvider"]
