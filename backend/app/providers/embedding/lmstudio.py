"""
LM Studio compatible embeddings provider.
"""

from typing import List, Dict, Any
import httpx
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from app.providers.embedding.registry import register_embedding_provider
from app.core.interfaces import EmbeddingProvider

__all__ = ["LMStudioEmbeddings", "LMStudioEmbeddingProvider"]


class LMStudioEmbeddings(Embeddings):
    """
    Custom embeddings class for LM Studio API compatibility.
    LM Studio only accepts string input, not token arrays.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "lm-studio"):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding


@register_embedding_provider("lmstudio")
class LMStudioEmbeddingProvider(EmbeddingProvider):
    """LM Studio embeddings provider."""

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def dimension(self) -> int:
        return 1024  # Default for text-embedding-bge-m3

    def create_embeddings(self, config: Dict[str, Any]) -> Embeddings:
        base_url = config.get("base_url", "http://localhost:1234/v1")
        model = config.get("model_name", "text-embedding-bge-m3")
        api_key = config.get("api_key", "lm-studio")

        return LMStudioEmbeddings(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "base_url" in config and "model_name" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_url": "http://localhost:1234/v1",
            "model_name": "text-embedding-bge-m3",
            "dimension": 1024,
        }
