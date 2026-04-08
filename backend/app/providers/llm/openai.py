"""
OpenAI LLM provider.
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from app.providers.llm.registry import register_llm_provider
from app.core.interfaces import LLMProvider

__all__ = ["OpenAIProvider"]


@register_llm_provider("openai")
class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
        ]

    def create_llm(self, config: Dict[str, Any]):
        model_name = config.get("model_name", "gpt-3.5-turbo")
        api_key = config.get("api_key", "")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 4096)
        base_url = config.get("base_url")

        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if base_url:
            kwargs["base_url"] = base_url

        return ChatOpenAI(**kwargs)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "model_name" in config or "api_key" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
