from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pathlib import Path
import shutil
import time
import asyncio

from app.models.schemas import (
    DocumentCreate, DocumentResponse, DocumentListResponse,
    CategoryResponse
)
from app.models.database import Document, Category
from app.services.database import get_db
from app.rag.document_loader import DocumentLoader, TextSplitter
from app.rag.vector_store import get_vector_store, DATA_DIR
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger("documents")


def get_or_create_default_category(db: Session) -> Category:
    default = db.query(Category).filter(Category.name == "Default").first()
    if not default:
        default = Category(name="Default", color="#6366f1")
        db.add(default)
        db.commit()
        db.refresh(default)
        logger.info("Created default category")
    return default


def process_document_background(doc_id: int, file_path: str, category_id: Optional[int]):
    """
    后台处理文档：解析、分块、向量化
    """
    from app.services.database import Database
    from app.models.database import Document as DocModel
    
    db_instance = Database()
    
    with db_instance.get_session() as db:
        try:
            doc = db.query(DocModel).filter(DocModel.id == doc_id).first()
            if not doc:
                logger.error(f"Document not found: {doc_id}")
                return
            
            doc.status = "processing"
            db.commit()
            logger.info(f"Document processing started: doc_id={doc_id}")
            
            loader = DocumentLoader()
            docs = loader.load_file(file_path)
            
            if not docs:
                doc.status = "failed"
                doc.error_message = "无法解析文档内容"
                db.commit()
                logger.warning(f"Failed to parse document: {doc_id}")
                return
            
            content = "\n\n".join([d.page_content for d in docs])
            doc.content = content
            db.commit()
            logger.info(f"Document content saved: doc_id={doc_id}, content_size={len(content)}")
            
            splitter = TextSplitter()
            chunks = splitter.split_documents(docs)
            
            logger.info(f"Document split into chunks: doc_id={doc_id}, chunk_count={len(chunks)}")
            
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["title"] = doc.title
                chunk.metadata["category_id"] = category_id
            
            vector_store = get_vector_store()
            vector_store.add_documents(chunks)
            
            doc.status = "completed"
            db.commit()
            logger.info(f"Document processed successfully: doc_id={doc_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            try:
                doc = db.query(DocModel).filter(DocModel.id == doc_id).first()
                if doc:
                    doc.status = "failed"
                    doc.error_message = str(e)
                    db.commit()
            except Exception as inner_e:
                logger.error(f"Failed to update document status: {inner_e}")


@router.post("/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db),
):
    start_time = time.time()
    logger.info(f"Creating document: title={document.title}, file_type={document.file_type}")
    
    category_id = document.category_id
    if category_id:
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            category_id = None
    
    if not category_id:
        default_cat = get_or_create_default_category(db)
        category_id = default_cat.id
    
    db_document = Document(
        title=document.title,
        file_path=document.file_path,
        file_type=document.file_type,
        content=document.content,
        category_id=category_id,
        status="completed" if document.content else "pending",
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    logger.info(f"Document saved to database: doc_id={db_document.id}")

    if document.content:
        vector_store = get_vector_store()
        from langchain_core.documents import Document as LangDocument
        doc = LangDocument(
            page_content=document.content,
            metadata={
                "doc_id": db_document.id,
                "title": db_document.title,
                "file_path": db_document.file_path,
                "file_type": db_document.file_type,
                "category_id": category_id,
            },
        )
        splitter = TextSplitter()
        chunks = splitter.split_documents([doc])
        
        try:
            vector_store.add_documents(chunks)
            logger.info(f"Document chunks added to vector store: doc_id={db_document.id}")
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            raise

    elapsed = time.time() - start_time
    logger.info(f"Document created successfully: doc_id={db_document.id}, elapsed_ms={elapsed * 1000:.2f}")
    return db_document


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    start_time = time.time()
    logger.info(f"Uploading document: filename={file.filename}, category_id={category_id}")
    
    upload_dir = DATA_DIR / "documents"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    file_size = file_path.stat().st_size
    logger.info(f"File saved to disk: filename={file.filename}, file_size={file_size}")

    file_type = Path(file.filename).suffix.lstrip(".")

    if category_id:
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            category_id = None
    
    if not category_id:
        default_cat = get_or_create_default_category(db)
        category_id = default_cat.id

    db_document = Document(
        title=Path(file.filename).stem,
        file_path=str(file_path),
        file_type=file_type,
        content=None,
        category_id=category_id,
        status="pending",
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    logger.info(f"Document record created: doc_id={db_document.id}, status=pending")

    background_tasks.add_task(
        process_document_background,
        db_document.id,
        str(file_path),
        category_id
    )

    elapsed = time.time() - start_time
    logger.info(f"Upload endpoint completed: doc_id={db_document.id}, elapsed_ms={elapsed * 1000:.2f}")
    return db_document


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    category_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Document)
    
    if category_id is not None:
        query = query.filter(Document.category_id == category_id)
    
    documents = query.offset(skip).limit(limit).all()
    total = query.count()
    return {"documents": documents, "total": total}


@router.get("/grouped")
async def list_documents_grouped(
    db: Session = Depends(get_db),
):
    categories = db.query(Category).all()
    
    result = []
    for cat in categories:
        docs = db.query(Document).filter(Document.category_id == cat.id).all()
        doc_responses = []
        for doc in docs:
            doc_responses.append(DocumentResponse(
                id=doc.id,
                title=doc.title,
                file_path=doc.file_path,
                file_type=doc.file_type,
                content=doc.content,
                category_id=doc.category_id,
                status=doc.status,
                error_message=doc.error_message,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                category=CategoryResponse(
                    id=cat.id,
                    name=cat.name,
                    color=cat.color,
                    created_at=cat.created_at,
                    document_count=len(docs),
                ) if cat else None,
            ))
        
        result.append({
            "category": CategoryResponse(
                id=cat.id,
                name=cat.name,
                color=cat.color,
                created_at=cat.created_at,
                document_count=len(docs),
            ),
            "documents": doc_responses,
        })
    
    uncategorized_docs = db.query(Document).filter(Document.category_id == None).all()
    if uncategorized_docs:
        doc_responses = []
        for doc in uncategorized_docs:
            doc_responses.append(DocumentResponse(
                id=doc.id,
                title=doc.title,
                file_path=doc.file_path,
                file_type=doc.file_type,
                content=doc.content,
                category_id=None,
                status=doc.status,
                error_message=doc.error_message,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                category=None,
            ))
        result.append({
            "category": None,
            "documents": doc_responses,
        })
    
    return result


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    document_update: dict,
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if "title" in document_update:
        document.title = document_update["title"]
    if "content" in document_update:
        document.content = document_update["content"]
    if "category_id" in document_update:
        if document_update["category_id"]:
            category = db.query(Category).filter(Category.id == document_update["category_id"]).first()
            if category:
                document.category_id = document_update["category_id"]
        else:
            document.category_id = None

    db.commit()
    db.refresh(document)
    return document


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    logger.info(f"Deleting document: document_id={document_id}")
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.warning(f"Document not found for deletion: document_id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        vector_store = get_vector_store()
        vector_store.delete(filter={"doc_id": document_id})
        logger.info(f"Document deleted from vector store: document_id={document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete from vector store: document_id={document_id}, error={e}")

    db.delete(document)
    db.commit()
    logger.info(f"Document deleted successfully: document_id={document_id}")
    return {"message": "Document deleted successfully"}


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document.status = "pending"
    db.commit()
    
    background_tasks.add_task(
        process_document_background,
        document.id,
        document.file_path,
        document.category_id
    )
    
    logger.info(f"Document reprocessing started: document_id={document_id}")
    return {"message": "Document reprocessing started", "document_id": document_id}


@router.post("/reindex")
async def reindex_documents(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    documents = db.query(Document).filter(Document.status == "completed").all()
    logger.info(f"Starting background reindex: document_count={len(documents)}")

    for doc in documents:
        doc.status = "pending"
    db.commit()

    def reindex_task():
        from app.services.database import Database
        from app.models.database import Document as DocModel
        
        db_instance = Database()
        with db_instance.get_session() as session:
            vector_store = get_vector_store()
            vector_store.reset()
            
            loader = DocumentLoader()
            splitter = TextSplitter()
            
            docs = session.query(DocModel).all()
            for doc in docs:
                if doc.file_path and Path(doc.file_path).exists():
                    try:
                        loaded_docs = loader.load_file(doc.file_path)
                        chunks = splitter.split_documents(loaded_docs)
                        
                        for chunk in chunks:
                            chunk.metadata["doc_id"] = doc.id
                            chunk.metadata["title"] = doc.title
                            chunk.metadata["category_id"] = doc.category_id
                        
                        vector_store.add_documents(chunks)
                        doc.status = "completed"
                        session.commit()
                    except Exception as e:
                        logger.error(f"Failed to reindex document {doc.id}: {e}")
                        doc.status = "failed"
                        doc.error_message = str(e)
                        session.commit()

    background_tasks.add_task(reindex_task)
    return {"message": f"Reindexing {len(documents)} documents in background"}
