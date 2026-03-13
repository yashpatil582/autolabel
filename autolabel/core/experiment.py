"""Iteration result dataclass for the autonomous loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IterationResult:
    """Captures the outcome of one autonomous loop iteration."""

    iteration: int
    strategy: str
    target_label: str
    new_lfs_generated: int
    new_lfs_valid: int
    f1_before: float
    f1_after: float
    f1_delta: float
    kept: bool
    active_lf_count: int
    coverage: float
    accuracy: float
    label_model_type: str = "generative"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "strategy": self.strategy,
            "target_label": self.target_label,
            "new_lfs_generated": self.new_lfs_generated,
            "new_lfs_valid": self.new_lfs_valid,
            "f1_before": self.f1_before,
            "f1_after": self.f1_after,
            "f1_delta": self.f1_delta,
            "kept": self.kept,
            "active_lf_count": self.active_lf_count,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
            "label_model_type": self.label_model_type,
            "error": self.error,
        }
