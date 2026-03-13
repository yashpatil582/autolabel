"""Token usage and cost tracking for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from autolabel.llm.base import LLMResponse


@dataclass
class CostEntry:
    """A single recorded LLM call with its cost metadata."""

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Accumulates token usage and estimated USD costs across LLM calls.

    Maintains a class-level pricing table (``COST_PER_1M_TOKENS``) mapping
    model names to ``(input_cost, output_cost)`` tuples expressed in USD per
    1 million tokens.
    """

    # (input_cost_per_1M, output_cost_per_1M) in USD
    COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
        # Anthropic
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-opus-4-20250514": (15.00, 75.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
        # OpenAI
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        # Groq (free tier / very cheap)
        "llama-3.1-8b-instant": (0.05, 0.08),
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "mixtral-8x7b-32768": (0.24, 0.24),
        # Local / Ollama models are free
        "llama3.1:latest": (0.0, 0.0),
        "llama3.1:8b": (0.0, 0.0),
        "llama3.1:70b": (0.0, 0.0),
        "mistral:7b": (0.0, 0.0),
    }

    def __init__(self) -> None:
        self.entries: list[CostEntry] = []

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute the estimated USD cost for a single call."""
        if model not in self.COST_PER_1M_TOKENS:
            return 0.0
        input_rate, output_rate = self.COST_PER_1M_TOKENS[model]
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    def record(self, response: LLMResponse) -> None:
        """Record token usage and cost from an LLMResponse.

        Args:
            response: The response returned by an LLM provider's generate call.
        """
        cost = self._compute_cost(response.model, response.input_tokens, response.output_tokens)
        entry = CostEntry(
            timestamp=datetime.now(tz=timezone.utc),
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
        )
        self.entries.append(entry)

    def total_cost(self) -> float:
        """Return the cumulative USD cost across all recorded calls."""
        return sum(e.cost_usd for e in self.entries)

    def total_tokens(self) -> dict[str, int]:
        """Return aggregate token counts.

        Returns:
            A dict with keys ``input_tokens``, ``output_tokens``, and
            ``total_tokens``.
        """
        input_tokens = sum(e.input_tokens for e in self.entries)
        output_tokens = sum(e.output_tokens for e in self.entries)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def summary(self) -> str:
        """Return a human-readable summary of accumulated usage and cost.

        Returns:
            A formatted multi-line string.
        """
        tokens = self.total_tokens()
        cost = self.total_cost()
        lines = [
            "LLM Usage Summary",
            "=" * 40,
            f"Total calls:        {len(self.entries)}",
            f"Input tokens:       {tokens['input_tokens']:,}",
            f"Output tokens:      {tokens['output_tokens']:,}",
            f"Total tokens:       {tokens['total_tokens']:,}",
            f"Estimated cost:     ${cost:.6f}",
        ]

        # Per-model breakdown when there are entries.
        if self.entries:
            models: dict[str, dict] = {}
            for entry in self.entries:
                key = f"{entry.provider}/{entry.model}"
                if key not in models:
                    models[key] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
                models[key]["calls"] += 1
                models[key]["input"] += entry.input_tokens
                models[key]["output"] += entry.output_tokens
                models[key]["cost"] += entry.cost_usd

            lines.append("")
            lines.append("Per-model breakdown:")
            lines.append("-" * 40)
            for key, stats in models.items():
                lines.append(
                    f"  {key}: {stats['calls']} calls, "
                    f"{stats['input'] + stats['output']:,} tokens, "
                    f"${stats['cost']:.6f}"
                )

        return "\n".join(lines)
