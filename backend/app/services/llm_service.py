"""
LLM Service with pluggable provider support via registry.
"""

from typing import Optional, Generator
import time
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config_manager import load_config
from app.core.logging_config import get_logger
from app.core.events import ConfigEvent
from app.services.base import BaseModelService

# Import providers to trigger registration via decorators
from app.providers.llm import LLMProviderRegistry

logger = get_logger("llm_service")


class LLMService(BaseModelService):
    """
    LLM Service with pluggable provider support.

    Uses the LLMProviderRegistry for provider creation and
    EventBus for hot-reload on configuration changes.
    """

    _subscribed_event_type = ConfigEvent.LLM_CONFIG_CHANGED
    _default_provider = "openai"

    def _load_config(self) -> None:
        """Load LLM configuration from config file."""
        config = load_config()
        self._config = config.get("models", {}).get("llm", {})
        self._provider = self._config.get("provider", self._default_provider)
        providers_config = self._config.get("providers", {})
        self._provider_config = providers_config.get(self._provider, {})

        base_url = self._provider_config.get("base_url", "")
        model_name = self._provider_config.get("model_name", "")
        logger.info(f"LLM config loaded: provider={self._provider}, base_url={base_url}, model={model_name}")

    def _create_instance(self) -> BaseChatModel:
        """Create LLM instance using registry."""
        logger.info(f"Creating LLM instance via registry: provider={self._provider}")

        try:
            return LLMProviderRegistry.create_llm(self._provider, self._provider_config)
        except ValueError as e:
            logger.warning(f"Provider creation failed: {e}, falling back to OpenAI")
            fallback_config = self._config.get("providers", {}).get("openai", {})
            return LLMProviderRegistry.create_llm("openai", fallback_config)

    def get_llm(self) -> BaseChatModel:
        """Get cached LLM instance, creating if necessary."""
        return self.get_instance()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response synchronously."""
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
        """Generate a streaming response."""
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
    """Get the global LLM service singleton."""
    global llm_service
    if llm_service is None:
        logger.debug("Creating new LLM service instance")
        llm_service = LLMService()
    return llm_service
