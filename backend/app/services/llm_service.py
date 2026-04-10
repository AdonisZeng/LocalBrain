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


def adaptive_k(question: str) -> int:
    """Determine retrieval k based on question complexity."""
    question_len = len(question)
    has_multiple = any(m in question for m in ["和", "与", "还是", "哪些", "多少", "分别"])
    has_comparison = any(c in question for c in ["比", "哪个更", "不如", "优于"])
    is_list_type = any(m in question for m in ["列出", "列举", "罗列", "有哪些"])

    if question_len > 100 or has_multiple or has_comparison:
        return 8
    elif question_len < 15 or is_list_type:
        return 2
    return 4


def expand_query(question: str, llm_service_instance) -> list[str]:
    """
    Use LLM to generate multiple search queries for the same question.
    Each query approaches the question from a different angle.
    """
    prompt = f"""为以下问题生成3个不同的搜索 query，用于从文档库中检索相关内容。
要求：每个 query 控制在20字以内，角度不同，用换行分隔。只需输出3行，不要其他内容。

问题：{question}

搜索 query："""
    try:
        response = llm_service_instance.generate(prompt)
        queries = [
            q.strip() for q in response.split('\n')
            if q.strip() and len(q.strip()) <= 25
        ]
        logger.debug(f"Query expansion: {len(queries)} queries generated")
        return queries[:3]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}, falling back to original question")
        return [question]


def hyde_expand(question: str, llm_service) -> list[str]:
    """
    HyDE (Hypothetical Document Embeddings):
    Generate a hypothetical answer document, then retrieve using both
    the original query and the generated answer. RRF fused.
    """
    hyde_prompt = f"""假设你是文档库中的内容。请为以下问题写一段假设性的回答，
这段回答应该是在相关文档中可能找到的内容格式和风格。
控制在100字以内。

问题：{question}

假设性回答："""
    try:
        hypothetical = llm_service.generate(hyde_prompt)
        logger.info(f"HyDE generated hypothetical answer: {len(hypothetical)} chars")
        return [question, hypothetical]
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}, falling back to original question")
        return [question]


def decompose_question(question: str, llm_service) -> list[str]:
    """
    Decompose a complex multi-hop question into sub-questions.
    Each sub-question is retrieved independently, then answers are synthesized.
    """
    decomp_prompt = f"""将以下复杂问题分解为2-3个简单的子问题。
每个子问题应该可以独立检索并回答。用换行分隔，仅输出子问题不要其他内容。

复杂问题：{question}

子问题："""
    try:
        response = llm_service.generate(decomp_prompt)
        sub_questions = [
            q.strip() for q in response.split('\n')
            if q.strip() and len(q.strip()) < 100
        ]
        if len(sub_questions) >= 2:
            logger.info(f"Question decomposed into {len(sub_questions)} sub-questions: {sub_questions}")
            return sub_questions
        logger.debug(f"Decomposition yielded {len(sub_questions)} sub-questions, using original")
        return [question]
    except Exception as e:
        logger.warning(f"Question decomposition failed: {e}")
        return [question]
