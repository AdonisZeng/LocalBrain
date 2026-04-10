from fastapi import APIRouter, Query, HTTPException
import time
import asyncio

from app.models.schemas import QAResult
from app.rag.vector_store import get_vector_store
from app.rag.utils import rrf_fuse
from app.services.llm_service import get_llm_service, adaptive_k, expand_query, hyde_expand, decompose_question
from app.core.logging_config import get_logger
from app.api.search import keyword_search as keyword_search_db
from app.core.config import get_config
from langchain_core.documents import Document

logger = get_logger("qa")

router = APIRouter()


async def _semantic_search_for_queries(
    vector_store, queries: list[str], k: int
) -> list[tuple[str, list[Document]]]:
    """Run similarity_search for multiple queries concurrently via asyncio.gather."""
    async def _single(q: str) -> tuple[str, list[Document]]:
        return (q, vector_store.similarity_search(q, k=k))

    results = await asyncio.gather(*[_single(q) for q in queries])
    return list(results)


async def _build_rrf_doc_map(
    vector_store, queries: list[str], k: int, rrf_k: int
) -> tuple[list[Document], dict[str, float]]:
    """Retrieve docs for multiple queries in parallel, fuse by RRF, return (docs, scores)."""
    doc_lists_raw = await _semantic_search_for_queries(vector_store, queries, k)

    # Build doc_id -> doc and accumulate RRF scores
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (q, docs) in enumerate(doc_lists_raw):
        for doc in docs:
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("file_path", f"q{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused_docs = [doc_map[doc_id] for doc_id in sorted_ids[:k] if doc_id in doc_map]
    return fused_docs, scores


async def hybrid_retrieve(query: str, k: int, rrf_k: int) -> list[Document]:
    """Hybrid retrieval: semantic + keyword with RRF fusion."""
    vector_store = get_vector_store()

    # Run semantic and keyword searches concurrently
    semantic_task = _semantic_search_for_queries(vector_store, [query], k * 2)
    keyword_task = keyword_search_db(query, k=k * 2)

    semantic_lists = await semantic_task
    keyword_results = await keyword_task

    # Build keyword doc map
    keyword_doc_map: dict[str, Document] = {}
    for rank, result in enumerate(keyword_results):
        doc_id = str(result.id)
        keyword_doc_map[doc_id] = Document(
            page_content=result.content,
            metadata={
                "doc_id": result.id,
                "title": result.title,
                "file_path": result.file_path,
                "file_type": result.file_type,
            }
        )

    # Combine semantic docs into same doc_map
    doc_map = {**keyword_doc_map}
    for rank, (_, docs) in enumerate(semantic_lists):
        for doc in docs:
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("file_path", f"sem_{rank}")
            doc_map[doc_id] = doc

    # RRF scores: keyword contribution
    scores: dict[str, float] = {doc_id: 1 / (rrf_k + rank + 1) for rank, doc_id in enumerate(keyword_doc_map)}

    # Semantic contribution
    for rank, (_, docs) in enumerate(semantic_lists):
        for doc in docs:
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("file_path", f"sem_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:k] if doc_id in doc_map]


@router.post("/", response_model=QAResult)
async def ask_question(
    question: str = Query(..., description="Question to ask"),
    mode: str = Query("semantic", description="Search mode: semantic, keyword, hybrid, hyde, or decompose"),
    k: int = Query(None, description="Number of context chunks to retrieve (default: adaptive)"),
):
    start_time = time.time()

    if k is None:
        k = adaptive_k(question)

    config = get_config()
    rrf_k = config.hybrid_search.get("rrf_k", 60)

    logger.info(f"Processing question: {question[:100]}..., mode={mode}, k={k}")

    vector_store = get_vector_store()
    llm_service = get_llm_service()

    try:
        search_start = time.time()
        if mode == "hybrid":
            docs = await hybrid_retrieve(question, k, rrf_k)
        elif mode == "keyword":
            keyword_results = await keyword_search_db(question, k=k)
            docs = [
                Document(
                    page_content=r.content,
                    metadata={"doc_id": r.id, "title": r.title, "file_path": r.file_path, "file_type": r.file_type}
                )
                for r in keyword_results
            ]
        elif mode == "hyde":
            queries = hyde_expand(question, llm_service)
            docs, _ = await _build_rrf_doc_map(vector_store, queries, k, rrf_k)
            logger.info(f"HyDE: {len(queries)} queries fused into {len(docs)} docs")
        elif mode == "decompose":
            sub_questions = decompose_question(question, llm_service)
            if len(sub_questions) > 1:
                docs, _ = await _build_rrf_doc_map(vector_store, sub_questions, k, rrf_k)
                logger.info(f"Decompose: {len(sub_questions)} sub-questions fused into {len(docs)} docs")
            else:
                docs = vector_store.similarity_search(question, k=k)
        elif mode == "semantic" and len(question) > 50:
            expanded = expand_query(question, llm_service)
            if len(expanded) > 1:
                docs, _ = await _build_rrf_doc_map(vector_store, expanded, k, rrf_k)
                logger.info(f"Query expansion: {len(expanded)} queries fused into {len(docs)} docs")
            else:
                docs = vector_store.similarity_search(question, k=k)
        else:
            docs = vector_store.similarity_search(question, k=k)

        search_elapsed = time.time() - search_start
        logger.info(f"Search complete: found {len(docs)} results in {search_elapsed * 1000:.2f}ms")

        if not docs:
            logger.warning(f"No relevant documents found for question: {question[:50]}")
            return QAResult(
                answer="抱歉，没有找到相关的文档来回答这个问题。",
                sources=[],
                question=question,
            )

        context = "\n\n".join([doc.page_content for doc in docs])
        logger.debug(f"Context prepared: size={len(context)}, chunks={len(docs)}")

        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "content": doc.page_content[:200],
                "file_path": doc.metadata.get("file_path", ""),
            }
            for doc in docs
        ]

        prompt = f"""基于以下文档内容，请回答用户的问题。如果文档中没有相关信息，请说明无法从已知文档中找到答案。

文档内容：
{context}

问题：{question}

请用简洁、准确的语言回答问题："""

        logger.info(f"Calling LLM service with prompt size={len(prompt)}")

        try:
            llm_start = time.time()
            answer = llm_service.generate(prompt)
            llm_elapsed = time.time() - llm_start
            logger.info(f"LLM response received: size={len(answer) if answer else 0} in {llm_elapsed * 1000:.2f}ms")
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"生成回答时发生错误：{str(e)}。请检查 LLM 配置是否正确。")

        total_elapsed = time.time() - start_time
        logger.info(f"Question processed successfully in {total_elapsed * 1000:.2f}ms")

        return QAResult(answer=answer, sources=sources, question=question)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理问题时发生错误：{str(e)}")
