"""
Abstract base classes (interfaces) for modular provider architecture.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this interface to add new LLM providers.

    Example:
        @register_llm_provider("myprovider")
        class MyProvider(LLMProvider):
            @property
            def name(self) -> str:
                return "myprovider"

            def create_llm(self, config: Dict[str, Any]) -> "BaseChatModel":
                # Return your LLM instance
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier (e.g., 'openai', 'ollama')."""
        pass

    @property
    def supported_models(self) -> List[str]:
        """List of supported model names. Empty means dynamic/unknown."""
        return []

    @abstractmethod
    def create_llm(self, config: Dict[str, Any]) -> "BaseChatModel":
        """
        Create and return an LLM instance based on config.

        Args:
            config: Provider-specific configuration dict with keys like:
                - base_url: API endpoint
                - model_name: Model identifier
                - api_key: API key (if needed)
                - temperature: Generation temperature
                - max_tokens: Max tokens to generate

        Returns:
            BaseChatModel instance (LangChain interface)
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate provider configuration.

        Default implementation returns True. Override to add validation.

        Args:
            config: Configuration dict to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this provider.

        Override to provide sensible defaults.
        """
        return {}


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implement this interface to add new embedding providers.

    Example:
        @register_embedding_provider("myembeddings")
        class MyEmbeddingProvider(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "myembeddings"

            @property
            def dimension(self) -> int:
                return 768

            def create_embeddings(self, config: Dict[str, Any]) -> "Embeddings":
                # Return your embeddings instance
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier (e.g., 'huggingface', 'openai')."""
        pass

    @property
    def dimension(self) -> int:
        """Default embedding dimension. Override in subclass."""
        return 768

    @property
    def dimension_env_var(self) -> Optional[str]:
        """Environment variable name for dimension override, if any."""
        return None

    @abstractmethod
    def create_embeddings(self, config: Dict[str, Any]) -> "Embeddings":
        """
        Create and return an embeddings instance based on config.

        Args:
            config: Provider-specific configuration dict with keys like:
                - base_url: API endpoint (for API-based providers)
                - model_name: Model identifier
                - device: Device to run on ('cpu', 'cuda')
                - dimension: Embedding dimension

        Returns:
            Embeddings instance (LangChain interface)
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate provider configuration.

        Default implementation returns True. Override to add validation.
        """
        return True

    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this provider.

        Override to provide sensible defaults.
        """
        return {}


class VectorStoreProvider(ABC):
    """
    Abstract base class for vector store providers.

    Implement this interface to add new vector store providers.

    Example:
        @register_vectorstore_provider("myvectorstore")
        class MyVectorStoreProvider(VectorStoreProvider):
            @property
            def name(self) -> str:
                return "myvectorstore"

            def create_vectorstore(
                self,
                embedding_function: "Embeddings",
                config: Dict[str, Any]
            ) -> "VectorStoreInterface":
                # Return your vector store implementation
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier (e.g., 'chroma', 'faiss', 'qdrant')."""
        pass

    @abstractmethod
    def create_vectorstore(
        self,
        embedding_function: "Embeddings",
        config: Dict[str, Any]
    ) -> "VectorStoreInterface":
        """
        Create and return a vector store instance.

        Args:
            embedding_function: Embeddings instance to use for encoding
            config: Provider-specific configuration dict with keys like:
                - persist_directory: Directory for persistence
                - collection_name: Name of the collection

        Returns:
            VectorStoreInterface implementation
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate provider configuration.

        Default implementation returns True. Override to add validation.
        """
        return True

    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this provider.

        Override to provide sensible defaults.
        """
        return {}


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector store operations.

    This defines the contract that all vector store implementations must follow.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List["Document"],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to process at a time

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List["Document"]:
        """
        Search for similar documents.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of matching Documents
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search for similar documents with scores.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
            filter: Filter dict for deletion
        """
        pass


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Implement this interface to add new document format loaders.

    Example:
        @register_document_loader
        class MyDocumentLoader(BaseDocumentLoader):
            @property
            def supported_extensions(self) -> List[str]:
                return [".myformat"]

            def load(self, file_path: str) -> List["Document"]:
                # Load and return documents
                ...
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions (e.g., ['.pdf', '.txt'])."""
        pass

    @property
    def priority(self) -> int:
        """
        Loader priority (higher = tried first).
        Default is 0. Override if a loader should be preferred.
        """
        return 0

    @abstractmethod
    def load(self, file_path: str) -> List["Document"]:
        """
        Load and return documents from file.

        Args:
            file_path: Path to the file to load

        Returns:
            List of LangChain Document objects
        """
        pass

    def can_load(self, file_path: str) -> bool:
        """
        Check if this loader can handle the file.

        Default implementation checks extension.
        Override for more complex detection.
        """
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions


# Type aliases for convenience
LLMProviderClass = type(LLMProvider)
EmbeddingProviderClass = type(EmbeddingProvider)
VectorStoreProviderClass = type(VectorStoreProvider)
DocumentLoaderClass = type(DocumentLoader)
