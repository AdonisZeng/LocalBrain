"""
Settings API endpoints with hot reload support.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.schemas import ModelSettingsResponse, LLMConfig, EmbeddingConfig
from app.core import config_manager
from app.core.events import EventBus, ConfigEvent

router = APIRouter(tags=["settings"])


class ModelSettingsUpdate(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig


class RagSettingsUpdate(BaseModel):
    parentDocument: dict
    compression: dict


@router.get("/models", response_model=ModelSettingsResponse)
async def get_model_settings():
    """Get current model settings."""
    try:
        settings = config_manager.get_model_settings()
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/models", response_model=ModelSettingsResponse)
async def update_model_settings(settings: ModelSettingsUpdate):
    """
    Update model settings and trigger hot reload.

    This endpoint saves the configuration and emits events
    to notify services that their configuration has changed.
    """
    try:
        config_manager.update_model_settings(
            llm_config=settings.llm.model_dump(),
            embedding_config=settings.embedding.model_dump()
        )

        # Emit events to trigger hot reload
        EventBus.emit_async(
            ConfigEvent.LLM_CONFIG_CHANGED,
            {"provider": settings.llm.provider}
        )
        EventBus.emit_async(
            ConfigEvent.EMBEDDING_CONFIG_CHANGED,
            {"provider": settings.embedding.provider}
        )

        return await get_model_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag")
async def get_rag_settings():
    """Get current RAG settings."""
    try:
        config = config_manager.get_config()
        return {
            "parentDocument": config.models.vectorstore.parent_document,
            "compression": config.models.vectorstore.compression,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rag")
async def update_rag_settings(settings: RagSettingsUpdate):
    """
    Update RAG settings and trigger hot reload.

    This endpoint saves the configuration and emits events
    to notify the vector store that its configuration has changed.
    """
    try:
        config_manager.update_rag_settings(
            parent_document=settings.parentDocument,
            compression=settings.compression,
        )

        # Emit event to trigger hot reload
        EventBus.emit_async(ConfigEvent.VECTORSTORE_CONFIG_CHANGED, {})

        return await get_rag_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
