"""Groq LLM provider (OpenAI-compatible API, serves open-source models)."""

from __future__ import annotations

import logging
import os
import time

from openai import OpenAI

from autolabel.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider(BaseLLMProvider):
    """LLM provider backed by Groq's fast inference API.

    Includes automatic retry with exponential backoff for rate limit errors.
    """

    def __init__(self, model: str = "", api_key: str = "") -> None:
        resolved_model = model or DEFAULT_MODEL
        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
        super().__init__(model=resolved_model, api_key=resolved_key)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_timeout_seconds: float | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Retry with backoff on rate limit errors
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout_seconds,
                )

                text = response.choices[0].message.content or ""
                usage = response.usage

                return LLMResponse(
                    text=text,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    model=self.model,
                    provider="groq",
                )
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate_limit" in error_str
                # Don't retry daily token limits — they won't reset soon
                is_daily = "tokens per day" in error_str or "tpd" in error_str
                if is_rate_limit and not is_daily and attempt < max_retries:
                    wait = 2**attempt  # 1s, 2s, 4s
                    logger.info("Groq rate limited, retrying in %ds...", wait)
                    time.sleep(wait)
                    continue
                raise
