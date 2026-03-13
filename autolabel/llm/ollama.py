"""Ollama (local) LLM provider."""

from __future__ import annotations

import ollama

from autolabel.llm.base import BaseLLMProvider, LLMResponse

DEFAULT_MODEL = "llama3.1:latest"


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a locally-running Ollama instance."""

    def __init__(self, model: str = "", api_key: str = "") -> None:
        resolved_model = model or DEFAULT_MODEL
        super().__init__(model=resolved_model, api_key=api_key)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a completion via the Ollama chat API.

        Args:
            prompt: The user prompt to send.
            system: An optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            An LLMResponse with the generated text and token usage.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
        )

        # Ollama returns token counts in the response metadata when available.
        input_tokens: int = 0
        output_tokens: int = 0
        if hasattr(response, "prompt_eval_count"):
            input_tokens = response.prompt_eval_count or 0
        elif isinstance(response, dict) and "prompt_eval_count" in response:
            input_tokens = response["prompt_eval_count"] or 0

        if hasattr(response, "eval_count"):
            output_tokens = response.eval_count or 0
        elif isinstance(response, dict) and "eval_count" in response:
            output_tokens = response["eval_count"] or 0

        # Extract the reply text.
        if isinstance(response, dict):
            text = response.get("message", {}).get("content", "")
        else:
            text = response.message.content or ""

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="ollama",
        )
