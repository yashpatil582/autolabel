"""Keep/discard logic for the autonomous loop."""

from __future__ import annotations


class Ratchet:
    """Decides whether to keep or discard new labeling functions.

    Keeps new LFs only if F1 improved by at least `min_improvement`.
    """

    def __init__(self, min_improvement: float = 0.005) -> None:
        self.min_improvement = min_improvement

    def should_keep(self, f1_before: float, f1_after: float) -> bool:
        """Return True if the improvement is sufficient to keep new LFs."""
        return (f1_after - f1_before) >= self.min_improvement
