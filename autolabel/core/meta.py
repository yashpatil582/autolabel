"""Meta-learning across iterations — tracks strategy effectiveness."""

from __future__ import annotations

import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetaLearner:
    """Tracks which (strategy, iteration_phase, coverage_bucket) combinations work best.

    Provides:
    - Success-rate lookup table for strategy/phase/coverage combinations
    - Adaptive temperature based on recent performance
    - Strategy weighting for biasing selection
    """

    def __init__(self, strategies: list[str]) -> None:
        self.strategies = strategies
        # (strategy, phase, coverage_bucket) -> [success_bool, ...]
        self._history: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
        # Per-strategy success tracking
        self._strategy_successes: dict[str, list[bool]] = defaultdict(list)
        self._recent_f1_deltas: list[float] = []
        self._base_temperature = 0.7
        self._stagnation_count = 0

    def update(
        self,
        strategy: str,
        iteration: int,
        coverage: float,
        kept: bool,
        f1_delta: float,
    ) -> None:
        """Record the outcome of an iteration."""
        phase = self._iteration_phase(iteration)
        coverage_bucket = self._coverage_bucket(coverage)
        key = (strategy, phase, coverage_bucket)

        self._history[key].append(kept)
        self._strategy_successes[strategy].append(kept)
        self._recent_f1_deltas.append(f1_delta)

        # Track stagnation
        if f1_delta <= 0:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

    def get_strategy_weights(self) -> dict[str, float]:
        """Return a weight for each strategy based on historical success rate.

        Higher weight = historically more effective. All weights sum to 1.0.
        """
        weights: dict[str, float] = {}

        for strategy in self.strategies:
            history = self._strategy_successes.get(strategy, [])
            if not history:
                weights[strategy] = 1.0  # No data — neutral weight
            else:
                # Success rate with a prior of 0.5 (Laplace smoothing)
                successes = sum(history)
                weights[strategy] = (successes + 1) / (len(history) + 2)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def suggest_strategy(self, current_f1: float, iteration: int) -> str | None:
        """Suggest a strategy based on meta-learning, or None for default behavior.

        Returns a weighted random choice biased toward historically effective strategies.
        """
        if len(self._recent_f1_deltas) < 3:
            return None  # Not enough data

        weights = self.get_strategy_weights()
        strategies = list(weights.keys())
        probs = list(weights.values())

        return random.choices(strategies, weights=probs, k=1)[0]

    def get_temperature(self) -> float:
        """Return adaptive temperature based on recent performance.

        - High temperature early (exploration)
        - Increase on stagnation (break out of rut)
        - Decrease on regression (be more conservative)
        """
        temp = self._base_temperature

        # Early exploration bonus
        if len(self._recent_f1_deltas) < 5:
            temp = 0.8

        # Stagnation: increase temperature to explore
        if self._stagnation_count >= 3:
            temp = min(temp + 0.1 * (self._stagnation_count - 2), 1.0)

        # Recent regression: decrease temperature
        recent = self._recent_f1_deltas[-3:]
        if len(recent) >= 3 and all(d < 0 for d in recent):
            temp = max(temp - 0.2, 0.3)

        return temp

    def get_success_rate(self, strategy: str, iteration: int, coverage: float) -> float:
        """Look up historical success rate for a (strategy, phase, coverage) combination."""
        phase = self._iteration_phase(iteration)
        coverage_bucket = self._coverage_bucket(coverage)
        key = (strategy, phase, coverage_bucket)

        history = self._history.get(key, [])
        if not history:
            return 0.5  # No data — neutral prior
        return sum(history) / len(history)

    @staticmethod
    def _iteration_phase(iteration: int) -> str:
        """Classify iteration into a phase."""
        if iteration <= 5:
            return "early"
        elif iteration <= 15:
            return "mid"
        else:
            return "late"

    @staticmethod
    def _coverage_bucket(coverage: float) -> str:
        """Bucket coverage into categories."""
        if coverage < 0.3:
            return "low"
        elif coverage < 0.7:
            return "mid"
        else:
            return "high"
