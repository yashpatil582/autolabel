"""Registry for tracking active and retired labeling functions."""

from __future__ import annotations

import logging
from typing import Sequence

from autolabel.lf.base import LabelingFunction

logger = logging.getLogger(__name__)


class LFRegistry:
    """Central bookkeeper for labeling functions across iterations.

    Maintains two pools:

    * **active** -- LFs currently contributing votes to the label matrix.
    * **retired** -- LFs removed during pruning but kept for audit / analysis.
    """

    def __init__(self) -> None:
        self.active_lfs: list[LabelingFunction] = []
        self.retired_lfs: list[LabelingFunction] = []
        self._name_index: dict[str, LabelingFunction] = {}

    # ------------------------------------------------------------------ #
    # Adding
    # ------------------------------------------------------------------ #

    def add(self, lf: LabelingFunction) -> None:
        """Add a single labeling function to the active pool.

        If an LF with the same name already exists, a numeric suffix is
        appended to make it unique.
        """
        if lf.name in self._name_index:
            base = lf.name
            for i in range(2, 1000):
                candidate = f"{base}_v{i}"
                if candidate not in self._name_index:
                    lf.name = candidate
                    break
            logger.debug("Renamed duplicate LF %s -> %s", base, lf.name)
        self.active_lfs.append(lf)
        self._name_index[lf.name] = lf
        logger.debug("Registered active LF: %s", lf.name)

    def add_batch(self, lfs: Sequence[LabelingFunction]) -> None:
        """Add multiple labeling functions at once."""
        for lf in lfs:
            self.add(lf)

    # ------------------------------------------------------------------ #
    # Retiring
    # ------------------------------------------------------------------ #

    def retire(self, lf_name: str) -> None:
        """Move an active LF to the retired pool by name.

        Raises:
            KeyError: If no active LF with the given name exists.
        """
        if lf_name not in self._name_index:
            raise KeyError(
                f"No active LF named '{lf_name}' found in the registry"
            )

        lf = self._name_index.pop(lf_name)
        self.active_lfs.remove(lf)
        self.retired_lfs.append(lf)
        logger.debug("Retired LF: %s", lf_name)

    def retire_batch(self, lf_names: Sequence[str]) -> None:
        """Retire multiple labeling functions at once."""
        for name in lf_names:
            self.retire(name)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_active(self) -> list[LabelingFunction]:
        """Return a shallow copy of the active LF list."""
        return list(self.active_lfs)

    def stats(self) -> dict[str, int | dict[str, int]]:
        """Return summary statistics about the registry.

        Returns:
            A dictionary with counts of active and retired LFs, as well
            as breakdowns by strategy and target label.
        """
        strategy_counts: dict[str, int] = {}
        label_counts: dict[str, int] = {}
        for lf in self.active_lfs:
            strategy_counts[lf.strategy] = (
                strategy_counts.get(lf.strategy, 0) + 1
            )
            label_counts[lf.target_label] = (
                label_counts.get(lf.target_label, 0) + 1
            )

        return {
            "total_active": len(self.active_lfs),
            "total_retired": len(self.retired_lfs),
            "by_strategy": strategy_counts,
            "by_label": label_counts,
        }

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.active_lfs)

    def __repr__(self) -> str:
        return (
            f"LFRegistry(active={len(self.active_lfs)}, "
            f"retired={len(self.retired_lfs)})"
        )
