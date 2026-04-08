"""
OpenAI embeddings provider.
"""

from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from app.providers.embedding.registry import register_embedding_provider
from app.core.interfaces import EmbeddingProvider

__all__ = ["OpenAIEmbeddingProvider"]


@register_embedding_provider("openai")
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return 1536  # Default for text-embedding-3-small

    def create_embeddings(self, config: Dict[str, Any]) -> OpenAIEmbeddings:
        api_key = config.get("api_key", "")
        model_name = config.get("model_name", "text-embedding-3-small")
        base_url = config.get("base_url")

        kwargs = {
            "model": model_name,
            "api_key": api_key,
        }

        if base_url:
            kwargs["base_url"] = base_url

        return OpenAIEmbeddings(**kwargs)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "api_key" in config or "model_name" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "text-embedding-3-small",
            "dimension": 1536,
        }
