"""
Custom OpenAI-compatible embeddings provider.
"""

from typing import List, Dict, Any
import httpx
from langchain_openai import OpenAIEmbeddings
from app.providers.embedding.registry import register_embedding_provider
from app.core.interfaces import EmbeddingProvider

__all__ = ["CustomEmbeddingProvider"]


@register_embedding_provider("custom")
class CustomEmbeddingProvider(EmbeddingProvider):
    """Custom OpenAI-compatible embeddings provider."""

    @property
    def name(self) -> str:
        return "custom"

    @property
    def dimension(self) -> int:
        return 768  # Common default

    def create_embeddings(self, config: Dict[str, Any]) -> OpenAIEmbeddings:
        base_url = config.get("base_url", "")
        model_name = config.get("model_name", "embedding-model")
        api_key = config.get("api_key", "custom")

        if not base_url:
            raise ValueError("Custom provider requires base_url in config")

        return OpenAIEmbeddings(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "base_url" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_url": "",
            "model_name": "embedding-model",
            "dimension": 768,
        }
