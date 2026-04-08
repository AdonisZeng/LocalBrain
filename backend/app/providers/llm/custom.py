"""
Custom/OpenAI-compatible LLM provider.
"""

from typing import Dict, Any, List
import httpx
from langchain_openai import ChatOpenAI
from app.providers.llm.registry import register_llm_provider
from app.core.interfaces import LLMProvider

__all__ = ["CustomProvider"]


@register_llm_provider("custom")
class CustomProvider(LLMProvider):
    """Custom OpenAI-compatible API endpoint provider."""

    @property
    def name(self) -> str:
        return "custom"

    @property
    def supported_models(self) -> List[str]:
        # Dynamic, depends on the custom endpoint
        return []

    def create_llm(self, config: Dict[str, Any]):
        base_url = config.get("base_url", "")
        model_name = config.get("model_name", "gpt-3.5-turbo")
        api_key = config.get("api_key", "custom")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 4096)

        if not base_url:
            raise ValueError("Custom provider requires base_url in config")

        return ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            http_client=httpx.Client(trust_env=False),
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "base_url" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_url": "",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
