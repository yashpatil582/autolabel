"""Base classes for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured response from an LLM provider call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


class BaseLLMProvider(ABC):
    """Abstract base class that all LLM providers must implement."""

    def __init__(self, model: str, api_key: str = "") -> None:
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            prompt: The user prompt to send.
            system: An optional system prompt.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum number of tokens to generate.
            request_timeout_seconds: Optional per-request timeout.

        Returns:
            An LLMResponse containing the generated text and usage metadata.
        """

    def generate_structured(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        request_timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Generate a deterministic completion suitable for structured output.

        Delegates to `generate` with temperature forced to 0.0 so the output
        is as reproducible as possible.

        Args:
            prompt: The user prompt to send.
            system: An optional system prompt.
            temperature: Sampling temperature (defaults to 0.0).
            request_timeout_seconds: Optional per-request timeout.

        Returns:
            An LLMResponse containing the generated text and usage metadata.
        """
        return self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
        )
