from fastapi import APIRouter, Depends, Query
from typing import Optional
import time

from app.models.schemas import QAResult
from app.rag.vector_store import get_vector_store
from app.services.llm_service import get_llm_service
from app.core.logging_config import get_logger

logger = get_logger("qa")

router = APIRouter()


@router.post("/", response_model=QAResult)
async def ask_question(
    question: str = Query(..., description="Question to ask"),
    mode: str = Query("semantic", description="Search mode: semantic, keyword, or hybrid"),
    k: int = Query(4, description="Number of context chunks to retrieve"),
):
    start_time = time.time()
    logger.info(f"Processing question: {question[:100]}...")
    
    vector_store = get_vector_store()
    llm_service = get_llm_service()

    try:
        search_start = time.time()
        docs = vector_store.similarity_search(question, k=k)
        search_elapsed = time.time() - search_start
        logger.info(f"Vector search complete: found {len(docs)} results in {search_elapsed * 1000:.2f}ms")

        if not docs:
            logger.warning(f"No relevant documents found for question: {question[:50]}")
            return QAResult(
                answer="抱歉，没有找到相关的文档来回答这个问题。",
                sources=[],
                question=question,
            )

        context = "\n\n".join([doc.page_content for doc in docs])
        context_size = len(context)
        logger.debug(f"Context prepared: size={context_size}, chunks={len(docs)}")

        sources = []
        for doc in docs:
            sources.append({
                "title": doc.metadata.get("title", "Untitled"),
                "content": doc.page_content[:200],
                "file_path": doc.metadata.get("file_path", ""),
            })

        prompt = f"""基于以下文档内容，请回答用户的问题。如果文档中没有相关信息，请说明无法从已知文档中找到答案。

文档内容：
{context}

问题：{question}

请用简洁、准确的语言回答问题："""

        prompt_size = len(prompt)
        logger.info(f"Calling LLM service with prompt size={prompt_size}")
        
        try:
            llm_start = time.time()
            answer = llm_service.generate(prompt)
            llm_elapsed = time.time() - llm_start
            answer_size = len(answer) if answer else 0
            logger.info(f"LLM response received: size={answer_size} in {llm_elapsed * 1000:.2f}ms")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = f"生成回答时发生错误：{str(e)}。请检查 LLM 配置是否正确。 "

        total_elapsed = time.time() - start_time
        logger.info(f"Question processed successfully in {total_elapsed * 1000:.2f}ms")
        
        return QAResult(
            answer=answer,
            sources=sources,
            question=question,
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return QAResult(
            answer=f"处理问题时发生错误：{str(e)}",
            sources=[],
            question=question,
        )
