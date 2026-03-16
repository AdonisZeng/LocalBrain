from typing import Optional, Generator
import httpx
import time
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config_manager import load_config
from app.core.logging_config import get_logger

logger = get_logger("llm_service")


class LLMService:
    def __init__(self):
        self._llm: Optional[BaseChatModel] = None
        self._config: Optional[dict] = None
        self._load_config()

    def _load_config(self):
        config = load_config()
        self._config = config.get("models", {}).get("llm", {})
        provider = self._config.get("provider", "unknown")
        providers_config = self._config.get("providers", {})
        provider_config = providers_config.get(provider, {})
        base_url = provider_config.get("base_url", "")
        model_name = provider_config.get("model_name", "")
        logger.info(f"LLM config loaded: provider={provider}, base_url={base_url}, model={model_name}")

    def _create_llm(self) -> BaseChatModel:
        provider = self._config.get("provider", "openai")
        providers_config = self._config.get("providers", {})
        provider_config = providers_config.get(provider, {})

        base_url = provider_config.get("base_url", "")
        model_name = provider_config.get("model_name", "gpt-3.5-turbo")
        api_key = provider_config.get("api_key", "")
        
        logger.info(f"Creating LLM instance: provider={provider}, base_url={base_url}, model={model_name}")

        if provider == "lmstudio":
            logger.debug("Creating LM Studio ChatOpenAI client")
            http_client = httpx.Client(trust_env=False)
            return ChatOpenAI(
                base_url=base_url,
                model=model_name,
                api_key=api_key or "lm-studio",
                temperature=0.7,
                http_client=http_client,
            )

        elif provider == "ollama":
            logger.debug("Creating Ollama ChatOpenAI client")
            http_client = httpx.Client(trust_env=False)
            return ChatOpenAI(
                base_url=f"{base_url}/v1",
                model=model_name,
                api_key="ollama",
                temperature=0.7,
                http_client=http_client,
            )

        elif provider == "openai":
            logger.debug("Creating OpenAI ChatOpenAI client")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.7,
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            logger.debug("Creating Anthropic ChatAnthropic client")
            return ChatAnthropic(
                model=model_name,
                api_key=api_key,
                temperature=0.7,
            )

        elif provider == "custom":
            logger.debug("Creating Custom ChatOpenAI client")
            http_client = httpx.Client(trust_env=False)
            return ChatOpenAI(
                base_url=base_url,
                model=model_name,
                api_key=api_key or "custom",
                temperature=0.7,
                http_client=http_client,
            )

        else:
            logger.warning(f"Unknown provider '{provider}', falling back to OpenAI")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0.7,
            )

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            logger.debug("LLM instance not initialized, creating new instance")
            self._llm = self._create_llm()
        return self._llm

    def reload_config(self):
        logger.info("Reloading LLM config")
        self._load_config()
        self._llm = None
        logger.info("LLM config reloaded, instance cleared")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        start_time = time.time()
        prompt_size = len(prompt) if prompt else 0
        logger.debug(f"Starting LLM generation: prompt_size={prompt_size}, has_system_prompt={bool(system_prompt)}")
        
        llm = self.get_llm()
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = llm.invoke(messages)
            elapsed = time.time() - start_time
            response_size = len(response.content) if response.content else 0
            logger.info(f"LLM generation complete: response_size={response_size}, elapsed={elapsed * 1000:.2f}ms")
            return response.content
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        start_time = time.time()
        prompt_size = len(prompt) if prompt else 0
        logger.debug(f"Starting LLM streaming generation: prompt_size={prompt_size}")
        
        llm = self.get_llm()
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        chunk_count = 0
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    chunk_count += 1
                    yield chunk.content
            elapsed = time.time() - start_time
            logger.info(f"LLM streaming complete: chunk_count={chunk_count}, elapsed={elapsed * 1000:.2f}ms")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"LLM streaming failed: {e}")
            raise


llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global llm_service
    if llm_service is None:
        logger.debug("Creating new LLM service instance")
        llm_service = LLMService()
    return llm_service
