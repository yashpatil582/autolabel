"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import os

import anthropic

from autolabel.llm.base import BaseLLMProvider, LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(self, model: str = "", api_key: str = "") -> None:
        resolved_model = model or DEFAULT_MODEL
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        super().__init__(model=resolved_model, api_key=resolved_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a completion via the Anthropic Messages API.

        Args:
            prompt: The user prompt to send.
            system: An optional system prompt.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            An LLMResponse with the generated text and token usage.
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        message = self.client.messages.create(**kwargs)

        text = ""
        for block in message.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=self.model,
            provider="anthropic",
        )
