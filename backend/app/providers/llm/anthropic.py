"""
Anthropic Claude LLM provider.
"""

from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic
from app.providers.llm.registry import register_llm_provider
from app.core.interfaces import LLMProvider

__all__ = ["AnthropicProvider"]


@register_llm_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> List[str]:
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]

    def create_llm(self, config: Dict[str, Any]):
        model_name = config.get("model_name", "claude-3-sonnet-20240229")
        api_key = config.get("api_key", "")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 4096)

        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "model_name" in config or "api_key" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
