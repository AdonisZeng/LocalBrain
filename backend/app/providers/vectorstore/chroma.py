"""
Chroma vector store provider.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from langchain_community.vectorstores import Chroma
from app.providers.vectorstore.registry import register_vectorstore_provider
from app.core.interfaces import VectorStoreProvider, VectorStoreInterface
from app.core.logging_config import get_logger

logger = get_logger("chroma_provider")

__all__ = ["ChromaVectorStoreProvider"]


@register_vectorstore_provider("chroma")
class ChromaVectorStoreProvider(VectorStoreProvider):
    """Chroma vector store provider."""

    @property
    def name(self) -> str:
        return "chroma"

    def create_vectorstore(
        self,
        embedding_function,
        config: Dict[str, Any]
    ) -> VectorStoreInterface:
        """
        Create a Chroma vector store.

        Args:
            embedding_function: Embeddings instance
            config: Configuration with keys:
                - persist_directory: Directory to persist data
                - collection_name: Name of collection

        Returns:
            Chroma vector store instance
        """
        persist_directory = config.get("persist_directory", "./data/chroma_db")
        collection_name = config.get("collection_name", "documents")

        # Ensure directory exists
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating Chroma vector store: persist_directory={persist_directory}, collection={collection_name}")

        return Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_function,
            collection_name=collection_name,
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "persist_directory" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "persist_directory": "./data/chroma_db",
            "collection_name": "documents",
        }
