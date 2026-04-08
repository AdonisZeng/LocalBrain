"""
Ollama embeddings provider.
"""

from typing import List, Dict, Any
import httpx
from langchain_openai import OpenAIEmbeddings
from app.providers.embedding.registry import register_embedding_provider
from app.core.interfaces import EmbeddingProvider

__all__ = ["OllamaEmbeddingProvider"]


@register_embedding_provider("ollama")
class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embeddings provider."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def dimension(self) -> int:
        return 768  # Common default, but varies by model

    def create_embeddings(self, config: Dict[str, Any]) -> OpenAIEmbeddings:
        base_url = config.get("base_url", "http://localhost:11434")
        model_name = config.get("model_name", "nomic-embed-text")

        return OpenAIEmbeddings(
            base_url=f"{base_url}/v1",
            model=model_name,
            api_key="ollama",
            http_client=httpx.Client(trust_env=False),
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "base_url" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_url": "http://localhost:11434",
            "model_name": "nomic-embed-text",
            "dimension": 768,
        }
