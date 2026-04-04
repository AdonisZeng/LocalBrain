from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from app.core.logging_config import get_logger

logger = get_logger("document_loader")

_MARKDOWN_LOADER = None


def _get_markdown_loader(encoding: str = "utf-8"):
    global _MARKDOWN_LOADER
    if _MARKDOWN_LOADER is None:
        try:
            _MARKDOWN_LOADER = UnstructuredMarkdownLoader
        except NameError:
            _MARKDOWN_LOADER = TextLoader
    return _MARKDOWN_LOADER


def detect_encoding(file_path: str) -> str:
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence})")
            return encoding if encoding else 'utf-8'
    except ImportError:
        logger.debug("chardet not installed, using utf-8")
        return 'utf-8'
    except Exception as e:
        logger.warning(f"Failed to detect encoding: {e}, using utf-8")
        return 'utf-8'


def is_valid_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    
    printable_chars = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    total_chars = len(text)
    
    if total_chars == 0:
        return False
    
    ratio = printable_chars / total_chars
    return ratio > 0.8


class DocumentLoader:
    def __init__(self, encoding: str = "utf-8", fast_mode: bool = True):
        self.encoding = encoding
        self.fast_mode = fast_mode

    def load_file(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                return self._load_pdf(file_path)
            elif suffix in [".md", ".markdown"]:
                return self._load_markdown(file_path)
            elif suffix == ".txt":
                return self._load_text(file_path)
            elif suffix in [".doc", ".docx"]:
                return self._load_docx(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return []

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        加载 PDF 文件
        fast_mode=True: 使用 PyMuPDF 直接提取文本（快速）
        fast_mode=False: 使用 pymupdf4llm 转换为 Markdown（保留格式）
        """
        docs = []
        
        if self.fast_mode:
            docs = self._load_pdf_fast(file_path)
            if docs:
                return docs
        
        try:
            import pymupdf4llm
            
            page_chunks = pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
            )
            
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
        
        if not docs:
            docs = self._load_pdf_fast(file_path)
        
        if not docs:
            logger.warning(f"Failed to extract valid text from PDF: {file_path}")
        
        return docs

    def _load_pdf_fast(self, file_path: str) -> List[Document]:
        """
        使用 PyMuPDF 快速提取 PDF 文本（并行处理）
        """
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
        
        try:
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

    def _load_text(self, file_path: str) -> List[Document]:
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'utf-16']
        
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

    def _load_markdown(self, file_path: str) -> List[Document]:
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        
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
                            doc.metadata["file_type"] = "md"
                        
                        logger.info(f"Loaded markdown file: {file_path}, encoding: {encoding}")
                        return docs
                    else:
                        continue
                        
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Error with encoding {encoding}: {e}")
                continue
        
        logger.error(f"Failed to load markdown file: {file_path}")
        return []

    def _load_docx(self, file_path: str) -> List[Document]:
        try:
            from docx import Document as DocxDocument
            
            doc = DocxDocument(file_path)
            content = "\n".join([para.text for para in doc.paragraphs if para.text])
            
            if not is_valid_text(content):
                logger.warning(f"Invalid content in docx file: {file_path}")
                return []
            
            path = Path(file_path)
            result = [Document(
                page_content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_name": path.name,
                    "file_type": "docx",
                }
            )]
            
            logger.info(f"Loaded docx file: {file_path}")
            return result
            
        except ImportError:
            logger.warning("python-docx not installed, cannot load .docx files")
            return []
        except Exception as e:
            logger.error(f"Failed to load docx file {file_path}: {e}")
            return []

    def load_directory(self, directory: str, glob_pattern: str = "**/*") -> List[Document]:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return []

        all_docs = []
        supported_extensions = [".md", ".markdown", ".txt", ".pdf", ".docx"]

        for ext in supported_extensions:
            for file_path in path.glob(f"{glob_pattern}{ext}"):
                docs = self.load_file(str(file_path))
                all_docs.extend(docs)

        logger.info(f"Loaded {len(all_docs)} documents from {directory}")
        return all_docs


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        splitting_strategies: Optional[dict] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategies = splitting_strategies or {}

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        all_splits = []

        for doc in documents:
            file_type = doc.metadata.get("file_type", "txt")
            splits = self._split_by_type(doc, file_type)
            all_splits.extend(splits)

        logger.info(
            f"Split {len(documents)} documents into {len(all_splits)} chunks"
        )
        return all_splits

    def _split_by_type(self, doc: Document, file_type: str) -> List[Document]:
        if file_type in ["md", "markdown"]:
            return self._split_markdown(doc)
        elif file_type == "pdf":
            return self._split_pdf(doc)
        else:
            return self._split_text(doc)

    def _split_markdown(self, doc: Document) -> List[Document]:
        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        md_splits = markdown_splitter.split_text(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        final_splits = []
        for split in md_splits:
            if len(split.page_content) > self.chunk_size:
                sub_splits = text_splitter.split_documents([split])
                final_splits.extend(sub_splits)
            else:
                final_splits.append(split)

        for split in final_splits:
            split.metadata.update(doc.metadata)

        return final_splits

    def _split_pdf(self, doc: Document) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size or 800,
            chunk_overlap=self.chunk_overlap or 100,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )

        splits = text_splitter.split_documents([doc])
        for split in splits:
            split.metadata.update(doc.metadata)

        return splits

    def _split_text(self, doc: Document) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        splits = text_splitter.split_documents([doc])
        for split in splits:
            split.metadata.update(doc.metadata)

        return splits
