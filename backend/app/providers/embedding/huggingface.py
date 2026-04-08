"""
HuggingFace sentence-transformers embeddings provider.
"""

from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from app.providers.embedding.registry import register_embedding_provider
from app.core.interfaces import EmbeddingProvider

__all__ = ["HuggingFaceEmbeddingProvider"]


@register_embedding_provider("huggingface")
class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace sentence-transformers embeddings provider."""

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def dimension(self) -> int:
        return 384  # Default for all-MiniLM-L6-v2

    def create_embeddings(self, config: Dict[str, Any]) -> HuggingFaceEmbeddings:
        model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = config.get("device", "cpu")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True  # HuggingFace has sensible defaults

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "dimension": 384,
        }
