from fastapi import APIRouter, Depends, Query
from typing import Optional
from sqlalchemy.orm import Session

from app.models.schemas import SearchResponse, SearchResult
from app.models.database import Document
from app.services.database import get_db
from app.rag.vector_store import get_vector_store

router = APIRouter()


@router.get("/", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="Search query"),
    mode: str = Query("semantic", description="Search mode: semantic, keyword, or hybrid"),
    k: int = Query(4, description="Number of results to return"),
    db: Session = Depends(get_db),
):
    if mode == "semantic":
        results = await semantic_search(q, k)
    elif mode == "keyword":
        results = await keyword_search(q, k, db)
    elif mode == "hybrid":
        results = await hybrid_search(q, k, db)
    else:
        results = []

    return SearchResponse(
        results=results,
        total=len(results),
        query=q,
    )


async def semantic_search(query: str, k: int = 4) -> list[SearchResult]:
    vector_store = get_vector_store()

    try:
        docs = vector_store.similarity_search(query, k=k)

        results = []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id", 0)
            title = doc.metadata.get("title", "Untitled")

            results.append(
                SearchResult(
                    id=doc_id,
                    title=title,
                    file_path=doc.metadata.get("file_path", ""),
                    content=doc.page_content[:500],
                    score=1.0,
                    file_type=doc.metadata.get("file_type", ""),
                )
            )

        return results
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


async def keyword_search(query: str, k: int = 4, db: Session = None) -> list[SearchResult]:
    if db is None:
        return []

    query_lower = query.lower()

    documents = db.query(Document).filter(
        Document.title.ilike(f"%{query_lower}%") |
        Document.content.ilike(f"%{query_lower}%")
    ).limit(k).all()

    results = []
    for doc in documents:
        content_preview = doc.content[:500] if doc.content else ""

        score = 1.0
        if doc.title.lower().find(query_lower) != -1:
            score = 1.5

        results.append(
            SearchResult(
                id=doc.id,
                title=doc.title,
                file_path=doc.file_path,
                content=content_preview,
                score=score,
                file_type=doc.file_type,
            )
        )

    return results


async def hybrid_search(query: str, k: int = 4, db: Session = None) -> list[SearchResult]:
    semantic_results = await semantic_search(query, k=k * 2)
    keyword_results = await keyword_search(query, k=k * 2, db=db)

    seen_ids = set()
    merged_results = []

    for result in semantic_results:
        if result.id not in seen_ids:
            seen_ids.add(result.id)
            result.score = result.score * 0.6
            merged_results.append(result)

    for result in keyword_results:
        if result.id not in seen_ids:
            seen_ids.add(result.id)
            result.score = result.score * 0.4
            merged_results.append(result)

    merged_results.sort(key=lambda x: x.score, reverse=True)

    return merged_results[:k]
