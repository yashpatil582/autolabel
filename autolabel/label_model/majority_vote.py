from __future__ import annotations

import numpy as np

from autolabel.label_model.base import BaseLabelModel

ABSTAIN = -1


class MajorityVoteLabelModel(BaseLabelModel):
    """Aggregate labels via unweighted majority vote.

    For each sample, counts votes per class (ignoring abstains) and returns
    the class with the most votes.  Ties are broken by choosing the lower
    class index.  Samples with zero coverage receive label -1.
    """

    num_classes_: int

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fit(self, label_matrix: np.ndarray, num_classes: int) -> MajorityVoteLabelModel:
        """Store the number of classes (no real fitting needed)."""
        self.num_classes_ = num_classes
        return self

    def predict(self, label_matrix: np.ndarray) -> np.ndarray:
        """Return the majority-vote label for every sample."""
        vote_counts = self._vote_counts(label_matrix)  # (n, C)
        # No-coverage samples: every count is 0
        no_coverage = vote_counts.sum(axis=1) == 0

        # argmax already picks the first (lowest index) in case of tie
        preds = vote_counts.argmax(axis=1).astype(np.intp)
        preds[no_coverage] = ABSTAIN
        return preds

    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Normalised vote counts as probability estimates."""
        vote_counts = self._vote_counts(label_matrix).astype(np.float64)
        totals = vote_counts.sum(axis=1, keepdims=True)
        # Samples with no coverage get uniform distribution
        no_coverage = (totals == 0).ravel()
        totals[no_coverage] = 1.0  # avoid division by zero
        proba = vote_counts / totals
        proba[no_coverage] = 1.0 / self.num_classes_
        return proba

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vote_counts(self, label_matrix: np.ndarray) -> np.ndarray:
        """Build a (n_samples, num_classes) matrix of vote counts."""
        n_samples, n_lfs = label_matrix.shape
        counts = np.zeros((n_samples, self.num_classes_), dtype=np.int64)
        for c in range(self.num_classes_):
            counts[:, c] = (label_matrix == c).sum(axis=1)
        return counts
