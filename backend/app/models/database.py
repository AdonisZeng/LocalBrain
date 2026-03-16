from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table, Enum
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    color = Column(String(20), nullable=True, default="#6366f1")
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="category")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False, unique=True)
    file_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=True)
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    category = relationship("Category", back_populates="documents")


class Link(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    source_doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    target_doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    target_url = Column(String(1000), nullable=True)
    link_text = Column(String(500), nullable=True)
    link_type = Column(String(20), default="wikilink")
    created_at = Column(DateTime, default=datetime.utcnow)

    source_doc = relationship("Document", foreign_keys=[source_doc_id], backref="outgoing_links")
    target_doc = relationship("Document", foreign_keys=[target_doc_id], backref="incoming_links")
