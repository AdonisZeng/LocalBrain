from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from app.models.schemas import CategoryCreate, CategoryResponse
from app.models.database import Category, Document
from app.services.database import get_db
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger("categories")


@router.post("/", response_model=CategoryResponse)
async def create_category(
    category: CategoryCreate,
    db: Session = Depends(get_db),
):
    logger.info(f"Creating category: name={category.name}")
    
    existing = db.query(Category).filter(Category.name == category.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="分类名称已存在")
    
    db_category = Category(
        name=category.name,
        color=category.color,
    )
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    
    response = CategoryResponse(
        id=db_category.id,
        name=db_category.name,
        color=db_category.color,
        created_at=db_category.created_at,
        document_count=0,
    )
    logger.info(f"Category created: id={db_category.id}, name={db_category.name}")
    return response


@router.get("/", response_model=List[CategoryResponse])
async def list_categories(
    db: Session = Depends(get_db),
):
    categories = db.query(Category).all()
    
    result = []
    for cat in categories:
        doc_count = db.query(func.count(Document.id)).filter(
            Document.category_id == cat.id
        ).scalar()
        result.append(CategoryResponse(
            id=cat.id,
            name=cat.name,
            color=cat.color,
            created_at=cat.created_at,
            document_count=doc_count,
        ))
    
    return result


@router.get("/{category_id}", response_model=CategoryResponse)
async def get_category(
    category_id: int,
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="分类不存在")
    
    doc_count = db.query(func.count(Document.id)).filter(
        Document.category_id == category.id
    ).scalar()
    
    return CategoryResponse(
        id=category.id,
        name=category.name,
        color=category.color,
        created_at=category.created_at,
        document_count=doc_count,
    )


@router.put("/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: int,
    category_update: CategoryCreate,
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="分类不存在")

    if category.name == "Default":
        raise HTTPException(status_code=400, detail="默认分类不能修改")

    existing = db.query(Category).filter(
        Category.name == category_update.name,
        Category.id != category_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="分类名称已存在")

    category.name = category_update.name
    category.color = category_update.color

    db.commit()
    db.refresh(category)
    
    doc_count = db.query(func.count(Document.id)).filter(
        Document.category_id == category.id
    ).scalar()
    
    logger.info(f"Category updated: id={category.id}, name={category.name}")
    return CategoryResponse(
        id=category.id,
        name=category.name,
        color=category.color,
        created_at=category.created_at,
        document_count=doc_count,
    )


@router.delete("/{category_id}")
async def delete_category(
    category_id: int,
    db: Session = Depends(get_db),
):
    logger.info(f"Deleting category: category_id={category_id}")
    
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="分类不存在")

    if category.name == "Default":
        raise HTTPException(status_code=400, detail="默认分类不能删除")

    default_category = db.query(Category).filter(Category.name == "Default").first()
    if default_category:
        db.query(Document).filter(Document.category_id == category_id).update(
            {"category_id": default_category.id}
        )

    db.delete(category)
    db.commit()
    
    logger.info(f"Category deleted: id={category_id}")
    return {"message": "分类删除成功"}
