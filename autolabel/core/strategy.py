"""LLM-driven strategy selector for the autonomous loop."""

from __future__ import annotations

import json
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

STRATEGIES = [
    "keyword",
    "regex",
    "fuzzy",
    "semantic",
    "abbreviation",
    "negation",
    "context",
    "compositional",
]

_STRATEGY_SELECTION_PROMPT = """\
You are an expert at programmatic data labeling. You are building labeling functions (LFs) \
to classify text into labels.

Current state:
- Task: {task_description}
- Labels: {label_space}
- Current F1: {current_f1:.4f}
- Active LFs: {num_active_lfs}
- Iteration: {iteration}
- Language: {language}

Coverage by label (fraction of dev examples correctly covered):
{label_coverage}

Recent history (last 5 iterations):
{recent_history}

Available strategies:
- keyword: Exact string matching (names, product names, distinctive terms)
- regex: Pattern matching with re module
- fuzzy: Handle misspellings, partial matches, abbreviations
- semantic: Context clues, related terms, descriptions
- abbreviation: Short forms, codes, acronyms
- negation: Exclusion patterns ("NOT this label")
- context: Surrounding context, sentence structure
- compositional: Combine multiple signals in one function
{language_note}
Pick the BEST strategy and target label to improve F1. Focus on:
1. Labels with lowest coverage (most misclassified)
2. Strategies not recently tried
3. Strategies appropriate for the error patterns

Respond with ONLY valid JSON (no markdown, no explanation):
{{"strategy": "<strategy_name>", "target_label": "<label>", "reasoning": "<brief reason>"}}
"""

_LANGUAGE_NOTES: dict[str, str] = {
    "en": "",
    "hi": "\nNote: Text is in Hindi (Devanagari script). "
    "keyword and regex strategies work well with Devanagari. "
    "abbreviation strategy is less relevant for Hindi text.",
    "mr": "\nNote: Text is in Marathi (Devanagari script). "
    "keyword and regex strategies work well with Devanagari. "
    "abbreviation strategy is less relevant for Marathi text.",
}


class StrategySelector:
    """Selects the next (strategy, target_label) pair using LLM analysis."""

    def __init__(
        self,
        provider: Any,
        label_space: list[str],
        task_description: str,
        language: str = "en",
    ) -> None:
        self.provider = provider
        self.label_space = label_space
        self.task_description = task_description
        self.language = language

    def select(
        self,
        current_f1: float,
        num_active_lfs: int,
        iteration: int,
        label_coverage: dict[str, float],
        recent_history: list[dict],
    ) -> tuple[str, str]:
        """Select the best (strategy, target_label) pair.

        Falls back to random selection if LLM output can't be parsed.
        """
        # Format label coverage
        coverage_str = "\n".join(
            f"  {label}: {cov:.2%}" for label, cov in sorted(label_coverage.items())
        )

        # Format recent history
        if recent_history:
            history_str = "\n".join(
                f"  iter {h.get('iteration', '?')}: strategy={h.get('strategy', '?')}, "
                f"label={h.get('target_label', '?')}, kept={h.get('kept', '?')}, "
                f"delta={h.get('f1_delta', 0):+.4f}"
                for h in recent_history[-5:]
            )
        else:
            history_str = "  (no history yet)"

        language_note = _LANGUAGE_NOTES.get(self.language, "")

        prompt = _STRATEGY_SELECTION_PROMPT.format(
            task_description=self.task_description,
            label_space=", ".join(self.label_space),
            current_f1=current_f1,
            num_active_lfs=num_active_lfs,
            iteration=iteration,
            label_coverage=coverage_str,
            recent_history=history_str,
            language=self.language,
            language_note=language_note,
        )

        try:
            response = self.provider.generate_structured(prompt=prompt)
            result = json.loads(response.text.strip())
            strategy = result.get("strategy", "keyword")
            target_label = result.get("target_label", self.label_space[0])

            # Validate
            if strategy not in STRATEGIES:
                logger.warning("LLM returned invalid strategy '%s', falling back", strategy)
                strategy = random.choice(STRATEGIES)
            if target_label not in self.label_space:
                # Try fuzzy match
                for label in self.label_space:
                    if (
                        target_label.lower() in label.lower()
                        or label.lower() in target_label.lower()
                    ):
                        target_label = label
                        break
                else:
                    logger.warning("LLM returned invalid label '%s', falling back", target_label)
                    target_label = self._lowest_coverage_label(label_coverage)

            # Force diversity: if same label failed 2+ times recently, pick a different one
            recent_labels = [
                h.get("target_label") for h in recent_history[-3:] if not h.get("kept", False)
            ]
            if recent_labels.count(target_label) >= 2:
                logger.info("Forcing label diversity — '%s' failed recently", target_label)
                target_label = self._lowest_coverage_label(
                    {k: v for k, v in label_coverage.items() if k != target_label}
                )
                strategy = random.choice(STRATEGIES)

            # Force strategy diversity: if same strategy used 3+ times recently, rotate
            recent_strategies = [h.get("strategy") for h in recent_history[-3:]]
            if recent_strategies.count(strategy) >= 2:
                other = [s for s in STRATEGIES if s != strategy]
                strategy = random.choice(other)

            return strategy, target_label

        except Exception as e:
            logger.warning("Strategy selection failed (%s), using fallback", e)
            return self._fallback_select(label_coverage)

    def _fallback_select(self, label_coverage: dict[str, float]) -> tuple[str, str]:
        """Random strategy + lowest coverage label."""
        strategy = random.choice(STRATEGIES)
        target_label = self._lowest_coverage_label(label_coverage)
        return strategy, target_label

    def _lowest_coverage_label(self, label_coverage: dict[str, float]) -> str:
        """Return the label with the lowest coverage."""
        if not label_coverage:
            return random.choice(self.label_space)
        return min(label_coverage, key=label_coverage.get)
