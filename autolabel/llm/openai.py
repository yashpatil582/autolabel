"""OpenAI LLM provider."""

from __future__ import annotations

import os

import openai

from autolabel.llm.base import BaseLLMProvider, LLMResponse

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str = "", api_key: str = "") -> None:
        resolved_model = model or DEFAULT_MODEL
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(model=resolved_model, api_key=resolved_key)
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Generate a completion via the OpenAI Chat Completions API.

        Args:
            prompt: The user prompt to send.
            system: An optional system prompt.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            An LLMResponse with the generated text and token usage.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout_seconds,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            provider="openai",
        )
