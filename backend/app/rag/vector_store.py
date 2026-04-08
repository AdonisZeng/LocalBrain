"""
Vector Store implementation with pluggable provider support.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import sqlite3
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.embedding_service import get_embedding_service
from app.core.logging_config import get_logger
from app.core.config import get_config
from app.core.events import EventBus, ConfigEvent, Event
from app.providers.vectorstore.registry import VectorStoreProviderRegistry

logger = get_logger("vector_store")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class VectorStore:
    """
    Vector store with pluggable provider support.

    Wraps the underlying vector store (via registry) with additional
    functionality like parent-child retrieval and compression.
    """

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

        self._load_config()

        # Get embedding function via dependency injection
        self.embeddings = get_embedding_service().get_embeddings()

        # Create vector store via registry
        self.vectorstore = self._create_vectorstore()

        # Initialize parent store if enabled
        if self.parent_document_enabled:
            self._init_parent_store()

        # Register for config changes (hot reload)
        EventBus.subscribe(ConfigEvent.VECTORSTORE_CONFIG_CHANGED, self._on_config_changed)

        elapsed = time.time() - start_time
        logger.info(f"Vector store initialized in {elapsed * 1000:.2f}ms")

    def _load_config(self):
        """Load configuration from config file."""
        config = get_config()
        vectorstore_config = config.models.vectorstore

        self.provider_name = vectorstore_config.provider
        self.persist_directory = Path(vectorstore_config.persist_directory or self.persist_directory)

        # Parent document config
        parent_config = vectorstore_config.parent_document
        self.parent_document_enabled = parent_config.get("enabled", True)
        self.parent_chunk_size = parent_config.get("parent_chunk_size", 2000)
        self.child_chunk_size = parent_config.get("child_chunk_size", 400)
        logger.info(f"Parent document config: enabled={self.parent_document_enabled}, parent_chunk_size={self.parent_chunk_size}, child_chunk_size={self.child_chunk_size}")

        # Compression config
        compression_config = vectorstore_config.compression
        self.compression_enabled = compression_config.get("enabled", True)
        self.compression_threshold = compression_config.get("threshold", 0.5)
        self.max_context_chars = compression_config.get("max_context_chars", 4000)
        logger.info(f"Compression config: enabled={self.compression_enabled}, threshold={self.compression_threshold}, max_context_chars={self.max_context_chars}")

    def _on_config_changed(self, event: Event) -> None:
        """Handle config change event from EventBus."""
        logger.info(f"VectorStore config change event received: {event.data}")
        self.reload_config()

    def reload_config(self) -> None:
        """Reload configuration."""
        self._load_config()
        logger.info("VectorStore config reloaded")

    def _get_embedding_dimension(self) -> int:
        """Get embedding model dimension by testing."""
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)

    def _get_existing_collection_dimension(self) -> int:
        """Get existing collection dimension from SQLite metadata."""
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
        """Create vector store via registry."""
        try:
            current_dimension = self._get_embedding_dimension()
            logger.info(f"Current embedding dimension: {current_dimension}")

            existing_dimension = self._get_existing_collection_dimension()
            logger.info(f"Existing collection dimension: {existing_dimension}")

            # Check for dimension mismatch
            if existing_dimension > 0 and existing_dimension != current_dimension:
                logger.warning(f"Dimension mismatch: existing={existing_dimension}, current={current_dimension}. Recreating collection.")
                client = Chroma(persist_directory=str(self.persist_directory))
                try:
                    client.delete_collection(name=self.collection_name)
                except Exception as e:
                    logger.debug(f"Failed to delete collection: {e}")

            # Use registry to create vector store
            config = {
                "persist_directory": str(self.persist_directory),
                "collection_name": self.collection_name,
            }

            logger.debug(f"Creating Chroma vector store via registry: config={config}")
            vectorstore = VectorStoreProviderRegistry.create_vectorstore(
                self.provider_name,
                self.embeddings,
                config
            )

            logger.info(f"Vector store created at {self.persist_directory}")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def _init_parent_store(self):
        """Initialize parent document store."""
        if not self.parent_document_enabled:
            return

        try:
            parent_dir = self.persist_directory / "parent"
            parent_dir.mkdir(parents=True, exist_ok=True)

            parent_config = {
                "persist_directory": str(parent_dir),
                "collection_name": self.parent_collection_name,
            }

            self.parent_vectorstore = VectorStoreProviderRegistry.create_vectorstore(
                self.provider_name,
                self.embeddings,
                parent_config
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
        """Add documents to vector store."""
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
                    # Process parent chunks for each document
                    for doc in batch:
                        doc_chunks = [doc]
                        parent_docs = self.parent_text_splitter.split_documents(doc_chunks)
                        for parent_doc in parent_docs:
                            parent_doc.metadata["doc_id"] = doc.metadata.get("doc_id")
                            parent_doc.metadata["title"] = doc.metadata.get("title")
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
        """Search for similar documents."""
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
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        use_parent_document: bool = True,
        use_compression: bool = True,
    ) -> List[tuple]:
        """Search for similar documents with scores."""
        start_time = time.time()
        query_size = len(query) if query else 0
        logger.debug(f"Starting similarity search with score: query_size={query_size}, k={k}")

        try:
            # Get base search results
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
                    results = self.parent_vectorstore.similarity_search_with_score(
                        query=query,
                        k=len(parent_doc_ids),
                        filter={"doc_id": {"$in": parent_doc_ids}},
                    )
                else:
                    results = [(doc, 0.0) for doc in child_docs]
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                )

            # Apply compression filtering
            if use_compression and self.compression_enabled and self.compression_threshold > 0 and results:
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
            logger.error(f"Similarity search with score failed: {e}", exc_info=True)
            raise

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None):
        """Delete documents from vector store."""
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
        """Get a retriever for this vector store."""
        logger.debug(f"Creating retriever with kwargs: {search_kwargs}")
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )

    def reset(self):
        """Reset the vector store - delete all data."""
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
    """Get the global vector store singleton."""
    global vector_store
    if vector_store is None:
        logger.debug("Creating new vector store instance")
        vector_store = VectorStore(
            persist_directory=str(DATA_DIR / "chroma_db"),
            collection_name="documents",
        )
    return vector_store
