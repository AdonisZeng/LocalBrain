"""
Shared utilities for document processing and retrieval.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


def is_valid_text(text: str) -> bool:
    """Check if text contains valid content (printable ratio > 80%)."""
    if not text or not text.strip():
        return False

    printable_chars = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    total_chars = len(text)

    if total_chars == 0:
        return False

    return (printable_chars / total_chars) > 0.8


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(2048)  # 2KB sample sufficient for detection
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            return encoding if encoding else 'utf-8'
    except ImportError:
        return 'utf-8'
    except Exception:
        return 'utf-8'


def rrf_fuse(
    result_lists: list[list[Tuple[str, Document, float]]],
    rrf_k: int = 60,
) -> list[Document]:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of (doc_id, doc, rank_weight) tuples per retrieval path.
        rrf_k: RRF smoothing parameter (default 60).

    Returns:
        Fused and sorted list of Documents.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for results in result_lists:
        for rank, (doc_id, doc, _) in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids if doc_id in doc_map]


def build_rrf_result_lists(
    doc_lists: list[tuple[str, list[Document]]],
    rrf_k: int = 60,
) -> list[Tuple[str, Document, float]]:
    """
    Build per-path result lists for RRF fusion from multiple retrieval paths.

    Args:
        doc_lists: List of (prefix, documents) tuples, e.g. [("hyde", docs), ("sem", docs)].
        rrf_k: RRF smoothing parameter.

    Returns:
        List of (doc_id, doc, rank) tuples suitable for rrf_fuse.
    """
    result: list[Tuple[str, Document, float]] = []
    for prefix, docs in doc_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("file_path", f"{prefix}_{rank}")
            result.append((doc_id, doc, rank))
    return result
