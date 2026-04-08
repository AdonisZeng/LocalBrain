# Embedding Providers package
# Import all providers to trigger @register_embedding_provider decorators
from app.providers.embedding import huggingface
from app.providers.embedding import lmstudio
from app.providers.embedding import ollama
from app.providers.embedding import openai
from app.providers.embedding import custom

from app.providers.embedding.registry import EmbeddingProviderRegistry, register_embedding_provider

__all__ = ["EmbeddingProviderRegistry", "register_embedding_provider"]
