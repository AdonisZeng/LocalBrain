"""
Base module for LLM providers.
Re-exports the abstract interface.
"""

from app.core.interfaces import LLMProvider

__all__ = ["LLMProvider"]
