"""
Document Loader Registry.
Implements the Service Provider pattern for pluggable document loaders.
"""

from typing import Dict, Type, List, Optional, Callable
from pathlib import Path
from langchain_core.documents import Document
from app.core.logging_config import get_logger

logger = get_logger("loader_registry")


class DocumentLoaderRegistry:
    """
    Registry for document loaders.

    Usage:
        # Register a loader (via decorator)
        @register_document_loader
        class MyLoader(BaseDocumentLoader):
            @property
            def supported_extensions(self):
                return [".myformat"]

            def load(self, file_path):
                ...

        # Get appropriate loader for a file
        loader = DocumentLoaderRegistry.get_loader("/path/to/file.pdf")
        docs = loader.load("/path/to/file.pdf")

        # List all supported extensions
        extensions = DocumentLoaderRegistry.list_extensions()
    """

    _loader_classes: Dict[str, Type] = {}
    _loader_instances: Dict[str, any] = {}

    @classmethod
    def register(cls, loader_class: Type) -> None:
        """
        Register a document loader class.

        Args:
            loader_class: Class implementing DocumentLoader interface

        Raises:
            TypeError: If loader_class doesn't have required interface
        """
        if not hasattr(loader_class, 'supported_extensions'):
            raise TypeError(f"{loader_class.__name__} must have supported_extensions property")

        if not hasattr(loader_class, 'load'):
            raise TypeError(f"{loader_class.__name__} must have load method")

        # Create instance to get extensions
        instance = loader_class()
        for ext in instance.supported_extensions:
            cls._loader_classes[ext] = loader_class
        logger.info(f"Registered document loader: {loader_class.__name__} for extensions: {instance.supported_extensions}")

    @classmethod
    def get_loader(cls, file_path: str):
        """
        Get appropriate loader for a file.

        Args:
            file_path: Path to the file

        Returns:
            DocumentLoader instance

        Raises:
            ValueError: If no loader registered for file extension
        """
        ext = Path(file_path).suffix.lower()
        loader_class = cls._loader_classes.get(ext)

        if loader_class is None:
            raise ValueError(f"No loader registered for extension: {ext}. Available: {cls.list_extensions()}")

        if loader_class not in cls._loader_instances:
            cls._loader_instances[loader_class] = loader_class()

        return cls._loader_instances[loader_class]

    @classmethod
    def list_extensions(cls) -> List[str]:
        """List all supported file extensions."""
        return list(cls._loader_classes.keys())

    @classmethod
    def list_loaders(cls) -> Dict[str, str]:
        """List all loaders by extension."""
        return {ext: cls._loader_classes[ext].__name__ for ext in cls._loader_classes}

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached loader instances. For testing."""
        cls._loader_instances.clear()
        logger.debug("Cleared document loader instances")


def register_document_loader(cls: Type) -> Type:
    """
    Decorator to register a document loader.

    Usage:
        @register_document_loader
        class MyLoader(BaseDocumentLoader):
            ...

    Args:
        cls: DocumentLoader class

    Returns:
        The same class (after registration)
    """
    DocumentLoaderRegistry.register(cls)
    return cls
