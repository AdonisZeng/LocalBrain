from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class CategoryBase(BaseModel):
    name: str
    color: Optional[str] = "#6366f1"


class CategoryCreate(CategoryBase):
    pass


class CategoryResponse(CategoryBase):
    id: int
    created_at: datetime
    document_count: int = 0

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    title: str
    file_path: str
    file_type: str


class DocumentCreate(DocumentBase):
    content: Optional[str] = None
    category_id: Optional[int] = None


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category_id: Optional[int] = None


class DocumentResponse(DocumentBase):
    id: int
    content: Optional[str] = None
    category_id: Optional[int] = None
    status: str = "pending"
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    category: Optional[CategoryResponse] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class SearchResult(BaseModel):
    id: int
    title: str
    file_path: str
    content: str
    score: float
    file_type: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str


class QAResult(BaseModel):
    answer: str
    sources: List[dict]
    question: str


class LinkBase(BaseModel):
    source_doc_id: int
    target_doc_id: Optional[int] = None
    target_url: Optional[str] = None
    link_text: Optional[str] = None
    link_type: str = "wikilink"


class LinkCreate(LinkBase):
    pass


class LinkResponse(LinkBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class LLMConfig(BaseModel):
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-3.5-turbo"
    api_key: str = ""


class EmbeddingConfig(BaseModel):
    provider: str = "huggingface"
    base_url: str = ""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384


class ModelSettingsResponse(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
