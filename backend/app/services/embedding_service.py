"""
Embedding Service with pluggable provider support via registry.
"""

from typing import List, Optional
import time
import asyncio
from langchain_core.embeddings import Embeddings

from app.core.config_manager import load_config
from app.core.logging_config import get_logger
from app.core.events import ConfigEvent
from app.services.base import BaseModelService

# Import providers to trigger registration via decorators
from app.providers.embedding import EmbeddingProviderRegistry

logger = get_logger("embedding_service")


class EmbeddingService(BaseModelService):
    """
    Embedding Service with pluggable provider support.

    Uses the EmbeddingProviderRegistry for provider creation and
    EventBus for hot-reload on configuration changes.
    """

    _subscribed_event_type = ConfigEvent.EMBEDDING_CONFIG_CHANGED
    _default_provider = "huggingface"

    def _load_config(self) -> None:
        """Load embedding configuration from config file."""
        config = load_config()
        self._config = config.get("models", {}).get("embedding", {})
        self._provider = self._config.get("provider", self._default_provider)
        providers_config = self._config.get("providers", {})
        self._provider_config = providers_config.get(self._provider, {})

        base_url = self._provider_config.get("base_url", "")
        model_name = self._provider_config.get("model_name", "")
        dimension = self._provider_config.get("dimension", 0)

        logger.info(f"Embedding config loaded: provider={self._provider}, base_url={base_url}, model={model_name}, dimension={dimension}")

    def _create_instance(self) -> Embeddings:
        """Create embeddings instance using registry."""
        logger.info(f"Creating embedding instance via registry: provider={self._provider}")

        try:
            return EmbeddingProviderRegistry.create_embeddings(self._provider, self._provider_config)
        except ValueError as e:
            logger.warning(f"Provider creation failed: {e}, falling back to HuggingFace")
            fallback_config = self._config.get("providers", {}).get("huggingface", {})
            return EmbeddingProviderRegistry.create_embeddings("huggingface", fallback_config)

    def get_embeddings(self) -> Embeddings:
        """Get cached embeddings instance, creating if necessary."""
        return self.get_instance()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        start_time = time.time()
        text_count = len(texts) if texts else 0
        logger.debug(f"Embedding {text_count} documents")

        embeddings = self.get_embeddings()
        try:
            result = embeddings.embed_documents(texts)
            elapsed = time.time() - start_time
            logger.info(f"Embedded {text_count} documents in {elapsed * 1000:.2f}ms")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to embed documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        start_time = time.time()
        text_size = len(text) if text else 0
        logger.debug(f"Embedding query of size {text_size}")

        embeddings = self.get_embeddings()
        try:
            result = embeddings.embed_query(text)
            elapsed = time.time() - start_time
            logger.info(f"Query embedded in {elapsed * 1000:.2f}ms")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to embed query: {e}")
            raise

    async def async_embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper for embed_documents to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def async_embed_query(self, text: str) -> List[float]:
        """Async wrapper for embed_query to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service singleton."""
    global embedding_service
    if embedding_service is None:
        logger.debug("Creating new embedding service instance")
        embedding_service = EmbeddingService()
    return embedding_service
