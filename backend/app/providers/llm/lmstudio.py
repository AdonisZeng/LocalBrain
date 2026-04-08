"""
LM Studio LLM provider.
"""

from typing import Dict, Any, List
import httpx
from langchain_openai import ChatOpenAI
from app.providers.llm.registry import register_llm_provider
from app.core.interfaces import LLMProvider

__all__ = ["LMStudioProvider"]


@register_llm_provider("lmstudio")
class LMStudioProvider(LLMProvider):
    """LM Studio local LLM provider."""

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def supported_models(self) -> List[str]:
        # Dynamic, depends on what's loaded in LM Studio
        return []

    def create_llm(self, config: Dict[str, Any]):
        base_url = config.get("base_url", "http://localhost:1234/v1")
        model_name = config.get("model_name", "qwen3.5-4b")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 4096)
        api_key = config.get("api_key", "lm-studio")

        return ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            http_client=httpx.Client(trust_env=False),
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "base_url" in config and "model_name" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_url": "http://localhost:1234/v1",
            "model_name": "qwen3.5-4b",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
