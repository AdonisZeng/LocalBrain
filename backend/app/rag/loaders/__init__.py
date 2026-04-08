# Document Loaders package
# Import all loaders to trigger @register_document_loader decorators
from app.rag.loaders import pdf_loader
from app.rag.loaders import markdown_loader
from app.rag.loaders import text_loader
from app.rag.loaders import docx_loader

from app.rag.loaders.registry import DocumentLoaderRegistry, register_document_loader

__all__ = ["DocumentLoaderRegistry", "register_document_loader"]
