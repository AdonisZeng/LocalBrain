"""
Ollama LLM provider.
"""

from typing import Dict, Any, List
import httpx
from langchain_openai import ChatOpenAI
from app.providers.llm.registry import register_llm_provider
from app.core.interfaces import LLMProvider

__all__ = ["OllamaProvider"]


@register_llm_provider("ollama")
class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supported_models(self) -> List[str]:
        # Dynamic, depends on what's installed in Ollama
        return []

    def create_llm(self, config: Dict[str, Any]):
        base_url = config.get("base_url", "http://localhost:11434")
        model_name = config.get("model_name", "llama3")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 4096)

        return ChatOpenAI(
            base_url=f"{base_url}/v1",
            model=model_name,
            api_key="ollama",
            temperature=temperature,
            max_tokens=max_tokens,
            http_client=httpx.Client(trust_env=False),
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "base_url" in config and "model_name" in config

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_url": "http://localhost:11434",
            "model_name": "llama3",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
