from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseLabelModel(ABC):
    """Base interface for label aggregation models.

    Label matrix format: shape (n_samples, n_lfs)
    - Values are indices into label_space (0, 1, 2, ...)
    - Value -1 means ABSTAIN
    """

    @abstractmethod
    def fit(self, label_matrix: np.ndarray, num_classes: int) -> BaseLabelModel:
        """Fit the model on a label matrix."""
        ...

    @abstractmethod
    def predict(self, label_matrix: np.ndarray) -> np.ndarray:
        """Predict labels. Returns array of shape (n_samples,) with class indices.
        Returns -1 for samples with no coverage."""
        ...

    @abstractmethod
    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Predict label probabilities. Returns (n_samples, num_classes)."""
        ...
