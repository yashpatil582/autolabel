"""LLM provider abstractions and factory."""

from __future__ import annotations

from autolabel.llm.anthropic import AnthropicProvider
from autolabel.llm.base import BaseLLMProvider, LLMResponse
from autolabel.llm.cost_tracker import CostEntry, CostTracker
from autolabel.llm.groq import GroqProvider
from autolabel.llm.ollama import OllamaProvider
from autolabel.llm.openai import OpenAIProvider

_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_provider(
    name: str,
    model: str = "",
    api_key: str = "",
) -> BaseLLMProvider:
    """Instantiate an LLM provider by name.

    Args:
        name: Provider identifier. One of ``"anthropic"``, ``"openai"``,
            or ``"ollama"``.
        model: Optional model override. Each provider has a sensible default.
        api_key: Optional API key. Providers fall back to environment variables
            when this is empty.

    Returns:
        A configured :class:`BaseLLMProvider` instance.

    Raises:
        ValueError: If *name* does not match a known provider.
    """
    name_lower = name.lower().strip()
    if name_lower not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {available}"
        )
    return _PROVIDERS[name_lower](model=model, api_key=api_key)


__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "CostEntry",
    "CostTracker",
    "LLMResponse",
    "GroqProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "get_provider",
]
