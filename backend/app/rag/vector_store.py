from typing import List, Optional, Dict, Any
from pathlib import Path
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import sqlite3

from app.services.embedding_service import get_embedding_service
from app.core.logging_config import get_logger
from app.core.config import get_config

logger = get_logger("vector_store")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        child_collection_name: str = "documents_child",
        parent_collection_name: str = "documents_parent",
    ):
        start_time = time.time()
        logger.info(f"Initializing vector store: persist_directory={persist_directory}, collection_name={collection_name}")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.child_collection_name = child_collection_name
        self.parent_collection_name = parent_collection_name

        self._load_parent_document_config()
        self._load_compression_config()
        
        logger.debug("Getting embedding service")
        self.embeddings = get_embedding_service().get_embeddings()
        
        self.vectorstore = self._create_vectorstore()
        
        if self.parent_document_enabled:
            self._init_parent_store()
        
        elapsed = time.time() - start_time
        logger.info(f"Vector store initialized in {elapsed * 1000:.2f}ms")

    def _load_parent_document_config(self):
        config = get_config()
        parent_config = config.models.vectorstore.parent_document
        self.parent_document_enabled = parent_config.get("enabled", True)
        self.parent_chunk_size = parent_config.get("parent_chunk_size", 2000)
        self.child_chunk_size = parent_config.get("child_chunk_size", 400)
        logger.info(f"Parent document config: enabled={self.parent_document_enabled}, parent_chunk_size={self.parent_chunk_size}, child_chunk_size={self.child_chunk_size}")

    def _load_compression_config(self):
        config = get_config()
        compression_config = config.models.vectorstore.compression
        self.compression_enabled = compression_config.get("enabled", True)
        self.compression_threshold = compression_config.get("threshold", 0.5)
        self.max_context_chars = compression_config.get("max_context_chars", 4000)
        logger.info(f"Compression config: enabled={self.compression_enabled}, threshold={self.compression_threshold}, max_context_chars={self.max_context_chars}")

    def _get_embedding_dimension(self) -> int:
        """
        获取嵌入模型的维度。
        """
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)

    def _get_existing_collection_dimension(self) -> int:
        """
        从 SQLite 数据库中获取现有集合的维度。
        """
        db_path = self.persist_directory / "chroma.sqlite3"
        if not db_path.exists():
            return 0
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT dimension FROM collections WHERE name = ?",
                (self.collection_name,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            return 0
        except Exception as e:
            logger.debug(f"Failed to get collection dimension from SQLite: {e}")
            return 0

    def _create_vectorstore(self) -> Chroma:
        try:
            logger.debug(f"Creating Chroma vectorstore: persist_directory={self.persist_directory}, collection_name={self.collection_name}")
            
            current_dimension = self._get_embedding_dimension()
            logger.info(f"Current embedding dimension: {current_dimension}")
            
            existing_dimension = self._get_existing_collection_dimension()
            logger.info(f"Existing collection dimension: {existing_dimension}")
            
            if existing_dimension > 0 and existing_dimension != current_dimension:
                logger.warning(f"Dimension mismatch: existing={existing_dimension}, current={current_dimension}. Recreating collection.")
                client = chromadb.PersistentClient(path=str(self.persist_directory))
                try:
                    client.delete_collection(name=self.collection_name)
                except Exception as e:
                    logger.debug(f"Failed to delete collection: {e}")
            
            vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            logger.info(f"Vector store created at {self.persist_directory}")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def _init_parent_store(self):
        if not self.parent_document_enabled:
            return
            
        try:
            parent_dir = self.persist_directory / "parent"
            parent_dir.mkdir(parents=True, exist_ok=True)
            
            self.parent_vectorstore = Chroma(
                persist_directory=str(parent_dir),
                embedding_function=self.embeddings,
                collection_name=self.parent_collection_name,
            )
            
            self.parent_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.parent_chunk_size,
                chunk_overlap=self.parent_chunk_size // 4,
                separators=["\n\n\n", "\n\n", "\n", " ", ""],
            )
            
            self.child_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.child_chunk_size,
                chunk_overlap=self.child_chunk_size // 4,
                separators=["\n\n\n", "\n\n", "\n", " ", ""],
            )
            
            logger.info(f"Parent document store initialized: {parent_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize parent store: {e}")
            self.parent_document_enabled = False

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        if not documents:
            logger.debug("No documents to add")
            return []
        
        start_time = time.time()
        doc_count = len(documents)
        logger.info(f"Adding {doc_count} documents to vector store (batch_size={batch_size})")

        all_ids = []
        total_batches = (doc_count + batch_size - 1) // batch_size
        
        try:
            for i in range(0, doc_count, batch_size):
                batch_start = time.time()
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches}: docs {i+1}-{min(i+batch_size, doc_count)}")
                
                ids = self.vectorstore.add_documents(batch)
                if ids:
                    all_ids.extend(ids)
                
                if self.parent_document_enabled and hasattr(self, 'parent_vectorstore'):
                    parent_docs = self.parent_text_splitter.split_documents(batch)
                    for parent_doc in parent_docs:
                        parent_doc.metadata["doc_id"] = batch[0].metadata.get("doc_id")
                        parent_doc.metadata["title"] = batch[0].metadata.get("title")
                    self.parent_vectorstore.add_documents(parent_docs)
                
                batch_elapsed = time.time() - batch_start
                total_elapsed = time.time() - start_time
                avg_time = total_elapsed / batch_num
                eta = avg_time * (total_batches - batch_num)
                
                logger.info(f"Batch {batch_num}/{total_batches} done in {batch_elapsed:.2f}s, total: {total_elapsed:.1f}s, ETA: {eta:.1f}s")
            
            elapsed = time.time() - start_time
            logger.info(f"Added {doc_count} documents to vector store in {elapsed:.2f}s ({doc_count/elapsed:.1f} docs/s)")
            return all_ids
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to add documents to vector store after {elapsed:.2f}s: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_parent_document: bool = True,
        use_compression: bool = True,
    ) -> List[Document]:
        start_time = time.time()
        query_size = len(query) if query else 0
        logger.debug(f"Starting similarity search: query_size={query_size}, k={k}, use_parent_document={use_parent_document}")

        try:
            if use_compression and self.compression_enabled and self.compression_threshold > 0:
                results = self.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                    use_parent_document=use_parent_document,
                    use_compression=use_compression,
                )
                return [doc for doc, score in results]
            
            if use_parent_document and self.parent_document_enabled and hasattr(self, 'parent_vectorstore'):
                child_docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                )
                
                if not child_docs:
                    return []
                
                parent_ids = set()
                parent_doc_ids = []
                for doc in child_docs:
                    doc_id = doc.metadata.get("doc_id")
                    if doc_id and doc_id not in parent_ids:
                        parent_ids.add(doc_id)
                        parent_doc_ids.append(doc_id)
                
                if parent_doc_ids:
                    parent_docs = self.parent_vectorstore.similarity_search(
                        query=query,
                        k=len(parent_doc_ids),
                        filter={"doc_id": {"$in": parent_doc_ids}},
                    )
                    
                    if parent_docs:
                        elapsed = time.time() - start_time
                        logger.info(f"Parent document search complete: found {len(parent_docs)} results in {elapsed * 1000:.2f}ms")
                        return parent_docs
                
                elapsed = time.time() - start_time
                logger.debug(f"No parent docs found, returning child docs: {len(child_docs)}")
                return child_docs
            else:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                )
                elapsed = time.time() - start_time
                logger.info(f"Similarity search complete: found {len(docs)} results in {elapsed * 1000:.2f}ms")
                return docs
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_parent_document: bool = True,
        use_compression: bool = True,
    ) -> List[tuple]:
        start_time = time.time()
        query_size = len(query) if query else 0
        logger.debug(f"Starting similarity search with score: query_size={query_size}, k={k}")

        try:
            docs = self.similarity_search(
                query=query,
                k=k,
                filter=filter,
                use_parent_document=use_parent_document,
                use_compression=False,
            )
            
            results = [(doc, 1.0 - (i * 0.1)) for i, doc in enumerate(docs)]
            
            if use_compression and self.compression_enabled and self.compression_threshold > 0:
                max_score = results[0][1] if results else 1.0
                min_score = results[-1][1] if results else 0.0
                score_range = max_score - min_score if max_score != min_score else 1.0
                
                filtered_results = []
                total_chars = 0
                
                for doc, score in results:
                    normalized_score = (max_score - score) / score_range if score_range > 0 else 0
                    
                    if normalized_score >= self.compression_threshold:
                        continue
                    
                    doc_chars = len(doc.page_content)
                    if total_chars + doc_chars > self.max_context_chars:
                        continue
                    
                    filtered_results.append((doc, score))
                    total_chars += doc_chars
                
                logger.info(f"Compression: filtered from {len(results)} to {len(filtered_results)} docs, chars: {total_chars}")
                results = filtered_results
            
            elapsed = time.time() - start_time
            logger.info(f"Similarity search with score complete: found {len(results)} results in {elapsed * 1000:.2f}ms")
            return results
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Similarity search with score failed: {e}")
            return []

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None):
        start_time = time.time()
        id_count = len(ids) if ids else 0
        logger.info(f"Deleting from vector store: id_count={id_count}")
        
        try:
            collection = self.vectorstore._collection
            if ids:
                collection.delete(ids=ids)
            elif filter:
                where_filter = None
                if "doc_id" in filter:
                    where_filter = {"doc_id": filter["doc_id"]}
                elif "document_id" in filter:
                    where_filter = {"document_id": filter["document_id"]}
                elif "source" in filter:
                    where_filter = {"source": filter["source"]}
                
                if where_filter:
                    collection.delete(where=where_filter)
                else:
                    logger.warning(f"Unsupported filter format: {filter}")
            else:
                logger.warning("No ids or filter provided for deletion")
                return
            
            elapsed = time.time() - start_time
            logger.info(f"Documents deleted from vector store in {elapsed * 1000:.2f}ms")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to delete from vector store: {e}")

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        logger.debug(f"Creating retriever with kwargs: {search_kwargs}")
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )

    def reset(self):
        start_time = time.time()
        logger.warning("Resetting vector store - this will delete all data")
        
        try:
            self.vectorstore.delete_collection()
            elapsed = time.time() - start_time
            logger.info(f"Vector store reset complete in {elapsed * 1000:.2f}ms")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to reset vector store: {e}")
            raise


vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global vector_store
    if vector_store is None:
        logger.debug("Creating new vector store instance")
        vector_store = VectorStore(
            persist_directory=str(DATA_DIR / "chroma_db"),
            collection_name="documents",
        )
    return vector_store
