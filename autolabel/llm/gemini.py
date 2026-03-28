"""Google Gemini LLM provider with multi-key rotation."""

from __future__ import annotations

import logging
import os

import google.generativeai as genai

from autolabel.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiProvider(BaseLLMProvider):
    """LLM provider backed by Google's Gemini API.

    Supports multiple API keys for rotation — pass comma-separated keys
    via the ``api_key`` argument or the ``GEMINI_API_KEY`` env var.
    When a key hits a rate limit, the provider rotates to the next one.
    """

    def __init__(self, model: str = "", api_key: str = "") -> None:
        resolved_model = model or DEFAULT_MODEL
        raw_keys = api_key or os.environ.get("GEMINI_API_KEY", "")
        # Support comma-separated keys for rotation
        self._keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self._key_index = 0

        super().__init__(model=resolved_model, api_key=self._keys[0] if self._keys else "")

        if not self._keys:
            raise ValueError("No Gemini API key provided. Set GEMINI_API_KEY or pass api_key.")

        logger.info(
            "GeminiProvider initialized with %d API key(s), model=%s",
            len(self._keys),
            resolved_model,
        )

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_timeout_seconds: float | None = None,
    ) -> LLMResponse:
        last_error = None

        # Try each key up to one full rotation
        for _ in range(len(self._keys)):
            current_key = self._keys[self._key_index]
            genai.configure(api_key=current_key)

            try:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system if system else None,
                )

                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": request_timeout_seconds}
                    if request_timeout_seconds
                    else None,
                )

                text = response.text or ""
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

                return LLMResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                    provider="gemini",
                )

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(
                    tok in error_str for tok in ("429", "rate", "quota", "resource_exhausted")
                )

                if is_rate_limit and len(self._keys) > 1:
                    logger.warning(
                        "Key %d/%d rate-limited, rotating to next key",
                        self._key_index + 1,
                        len(self._keys),
                    )
                    self._key_index = (self._key_index + 1) % len(self._keys)
                    last_error = e
                    continue
                else:
                    raise

        raise RuntimeError(f"All {len(self._keys)} Gemini API keys exhausted: {last_error}")
