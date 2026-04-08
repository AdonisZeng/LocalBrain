"""
Plain text document loader.
"""

from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from app.rag.loaders.registry import register_document_loader
from app.rag.utils import is_valid_text, detect_encoding
from app.core.logging_config import get_logger

logger = get_logger("text_loader")


@register_document_loader
class TextLoader:
    """Plain text document loader."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt"]

    def load(self, file_path: str) -> List[Document]:
        """Load text file with encoding detection."""
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'utf-16']

        # Add detected encoding if not in list
        detected_encoding = detect_encoding(file_path)
        if detected_encoding and detected_encoding.lower() not in [e.lower() for e in encodings_to_try]:
            encodings_to_try.insert(0, detected_encoding)

        for encoding in encodings_to_try:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                docs = loader.load()

                if docs and docs[0].page_content:
                    content = docs[0].page_content
                    if is_valid_text(content):
                        path = Path(file_path)
                        for doc in docs:
                            doc.metadata["file_path"] = str(file_path)
                            doc.metadata["file_name"] = path.name
                            doc.metadata["file_type"] = "txt"

                        logger.info(f"Loaded text file: {file_path}, encoding: {encoding}")
                        return docs
                    else:
                        logger.debug(f"Content invalid with encoding {encoding}")
                        continue

            except UnicodeDecodeError:
                logger.debug(f"Failed to decode with encoding {encoding}")
                continue
            except Exception as e:
                logger.debug(f"Error with encoding {encoding}: {e}")
                continue

        logger.error(f"Failed to load text file with any encoding: {file_path}")
        return []
