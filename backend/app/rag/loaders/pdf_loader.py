"""
PDF document loader with multiple fallback strategies.
"""

from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from app.rag.loaders.registry import register_document_loader
from app.rag.utils import is_valid_text
from app.core.logging_config import get_logger

logger = get_logger("pdf_loader")


@register_document_loader
class PDFLoader:
    """PDF document loader with fast and accurate modes."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    @property
    def priority(self) -> int:
        return 10  # Higher priority

    def load(self, file_path: str) -> List[Document]:
        """
        Load PDF file with multiple fallback strategies.

        Strategy order:
        1. pymupdf4llm (markdown-preserving)
        2. PyMuPDF fast mode (parallel extraction)
        3. pdfplumber
        4. PyPDFLoader
        """
        docs = []

        # Try pymupdf4llm first for better quality
        try:
            import pymupdf4llm
            page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)

            for chunk in page_chunks:
                text = chunk.get('text', '')
                if text and is_valid_text(text):
                    metadata = chunk.get('metadata', {})
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "page": metadata.get('page', 0),
                            "source": file_path,
                        }
                    ))

            if docs:
                path = Path(file_path)
                for doc in docs:
                    doc.metadata["file_path"] = str(file_path)
                    doc.metadata["file_name"] = path.name
                    doc.metadata["file_type"] = "pdf"
                logger.info(f"Loaded PDF with pymupdf4llm: {file_path}, pages: {len(docs)}")
                return docs
        except Exception as e:
            logger.debug(f"pymupdf4llm failed: {e}")

        # Fall back to PyMuPDF fast mode
        docs = self._load_fast(file_path)
        if docs:
            return docs

        # Fall back to pdfplumber
        docs = self._load_pdfplumber(file_path)
        if docs:
            return docs

        # Last resort: PyPDFLoader
        docs = self._load_pypdf(file_path)
        return docs

    def _load_fast(self, file_path: str) -> List[Document]:
        """Load PDF using PyMuPDF with parallel extraction."""
        docs = []
        fitz_doc = None

        try:
            import fitz
            fitz_doc = fitz.open(file_path)
            total_pages = len(fitz_doc)
            logger.info(f"Loading PDF with PyMuPDF (fast mode): {file_path}, total_pages: {total_pages}")

            def extract_page(page_num: int) -> Optional[Document]:
                try:
                    page = fitz_doc[page_num]
                    text = page.get_text()
                    if text and is_valid_text(text):
                        return Document(
                            page_content=text,
                            metadata={
                                "page": page_num + 1,
                                "source": file_path,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Failed to extract page {page_num}: {e}")
                return None

            num_workers = min(multiprocessing.cpu_count(), 8)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(extract_page, i): i for i in range(total_pages)}

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        docs.append(result)

            if docs:
                docs.sort(key=lambda x: x.metadata.get("page", 0))
                path = Path(file_path)
                for d in docs:
                    d.metadata["file_path"] = str(file_path)
                    d.metadata["file_name"] = path.name
                    d.metadata["file_type"] = "pdf"
                logger.info(f"Loaded PDF with PyMuPDF (fast): {file_path}, pages: {len(docs)}")
                return docs

        except Exception as e:
            logger.debug(f"PyMuPDF fast mode failed: {e}")
        finally:
            if fitz_doc is not None:
                fitz_doc.close()

        return docs

    def _load_pdfplumber(self, file_path: str) -> List[Document]:
        """Load PDF using pdfplumber."""
        docs = []

        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and is_valid_text(text):
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "page": page_num + 1,
                                "source": file_path,
                            }
                        ))

            if docs:
                path = Path(file_path)
                for d in docs:
                    d.metadata["file_path"] = str(file_path)
                    d.metadata["file_name"] = path.name
                    d.metadata["file_type"] = "pdf"
                logger.info(f"Loaded PDF with pdfplumber: {file_path}, pages: {len(docs)}")
                return docs

        except Exception as e:
            logger.debug(f"pdfplumber failed: {e}")

        return docs

    def _load_pypdf(self, file_path: str) -> List[Document]:
        """Load PDF using PyPDFLoader."""
        docs = []

        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()

            for d in raw_docs:
                if is_valid_text(d.page_content):
                    docs.append(d)

            if docs:
                path = Path(file_path)
                for d in docs:
                    d.metadata["file_path"] = str(file_path)
                    d.metadata["file_name"] = path.name
                    d.metadata["file_type"] = "pdf"
                logger.info(f"Loaded PDF with PyPDFLoader: {file_path}, pages: {len(docs)}")
                return docs

        except Exception as e:
            logger.debug(f"PyPDFLoader failed: {e}")

        return docs
