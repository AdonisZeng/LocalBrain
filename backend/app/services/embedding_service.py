from typing import Optional, List
import time
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import httpx
from openai import OpenAI

from app.core.config_manager import load_config
from app.core.logging_config import get_logger

logger = get_logger("embedding_service")


class LMStudioEmbeddings(Embeddings):
    """
    自定义嵌入类，用于兼容 LM Studio 的嵌入 API。
    LM Studio 只接受字符串格式的 input，不支持 token 数组。
    """

    def __init__(self, base_url: str, model: str, api_key: str = "lm-studio"):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对多个文档进行嵌入。
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询进行嵌入。
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding


class EmbeddingService:
    def __init__(self):
        self._embeddings: Optional[Embeddings] = None
        self._config: Optional[dict] = None
        self._load_config()

    def _load_config(self):
        config = load_config()
        self._config = config.get("models", {}).get("embedding", {})
        provider = self._config.get("provider", "huggingface")
        providers_config = self._config.get("providers", {})
        provider_config = providers_config.get(provider, {})
        base_url = provider_config.get("base_url", "")
        model_name = provider_config.get("model_name", "")
        logger.info(f"Embedding config loaded: provider={provider}, base_url={base_url}, model={model_name}")

    def _create_embeddings(self) -> Embeddings:
        provider = self._config.get("provider", "huggingface")
        providers_config = self._config.get("providers", {})
        provider_config = providers_config.get(provider, {})

        base_url = provider_config.get("base_url", "")
        model_name = provider_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info(f"Creating embedding instance: provider={provider}, base_url={base_url}, model={model_name}")

        if provider == "huggingface":
            logger.debug("Creating HuggingFace embeddings client")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        elif provider == "lmstudio":
            logger.debug("Creating LM Studio embeddings client")
            return LMStudioEmbeddings(
                base_url=base_url,
                model=model_name,
                api_key="lm-studio",
            )

        elif provider == "ollama":
            logger.debug("Creating Ollama OpenAIEmbeddings client")
            http_client = httpx.Client(trust_env=False)
            return OpenAIEmbeddings(
                base_url=f"{base_url}/v1",
                model=model_name,
                api_key="ollama",
                http_client=http_client,
            )

        elif provider == "openai":
            api_key = provider_config.get("api_key", "")
            logger.debug("Creating OpenAI embeddings client")
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
            )

        elif provider == "custom":
            api_key = provider_config.get("api_key", "")
            logger.debug("Creating Custom OpenAIEmbeddings client")
            http_client = httpx.Client(trust_env=False)
            return OpenAIEmbeddings(
                base_url=base_url,
                model=model_name,
                api_key=api_key or "custom",
                http_client=http_client,
            )

        else:
            logger.warning(f"Unknown provider '{provider}', falling back to HuggingFace")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            logger.debug("Embedding instance not initialized, creating new instance")
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def reload_config(self):
        logger.info("Reloading embedding config")
        self._load_config()
        self._embeddings = None
        logger.info("Embedding config reloaded, instance cleared")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start_time = time.time()
        text_count = len(texts) if texts else 0
        logger.debug(f"Embedding {text_count} documents")
        
        embeddings = self.get_embeddings()
        try:
            result = embeddings.embed_documents(texts)
            elapsed = time.time() - start_time
            logger.info(f"Embedded {text_count} documents in {elapsed * 1000:.2f}ms")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to embed documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        start_time = time.time()
        text_size = len(text) if text else 0
        logger.debug(f"Embedding query of size {text_size}")
        
        embeddings = self.get_embeddings()
        try:
            result = embeddings.embed_query(text)
            elapsed = time.time() - start_time
            logger.info(f"Query embedded in {elapsed * 1000:.2f}ms")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to embed query: {e}")
            raise


embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global embedding_service
    if embedding_service is None:
        logger.debug("Creating new embedding service instance")
        embedding_service = EmbeddingService()
    return embedding_service
