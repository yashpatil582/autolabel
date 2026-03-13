from __future__ import annotations

import numpy as np

from autolabel.label_model.base import BaseLabelModel

ABSTAIN = -1


class WeightedVoteLabelModel(BaseLabelModel):
    """Aggregate labels via accuracy-weighted voting.

    Per-LF weights are derived from how often each LF agrees with the
    majority of the *other* LFs (on samples where it does not abstain).
    The agreement rate is clipped to [0.5, 1.0] so that every active LF
    receives at least some positive weight.
    """

    num_classes_: int
    weights_: np.ndarray  # (n_lfs,)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fit(self, label_matrix: np.ndarray, num_classes: int) -> WeightedVoteLabelModel:
        """Compute per-LF accuracy weights from the label matrix."""
        self.num_classes_ = num_classes
        n_samples, n_lfs = label_matrix.shape
        weights = np.ones(n_lfs, dtype=np.float64)

        for j in range(n_lfs):
            active_mask = label_matrix[:, j] != ABSTAIN  # samples where LF j votes
            if active_mask.sum() == 0:
                weights[j] = 0.5
                continue

            # Majority vote of the *other* LFs for the active rows
            other_cols = np.delete(label_matrix, j, axis=1)  # (n, n_lfs-1)
            active_other = other_cols[active_mask]  # (n_active, n_lfs-1)
            active_labels_j = label_matrix[active_mask, j]  # (n_active,)

            majority_others = self._majority_of(active_other, num_classes)  # (n_active,)

            # Some rows may have no coverage from other LFs (-1); skip those
            valid = majority_others != ABSTAIN
            if valid.sum() == 0:
                weights[j] = 0.5
                continue

            agreement = (active_labels_j[valid] == majority_others[valid]).mean()
            weights[j] = np.clip(agreement, 0.5, 1.0)

        self.weights_ = weights
        return self

    def predict(self, label_matrix: np.ndarray) -> np.ndarray:
        """Return the weighted-vote label for every sample."""
        weighted = self._weighted_counts(label_matrix)
        no_coverage = weighted.sum(axis=1) == 0.0

        preds = weighted.argmax(axis=1).astype(np.intp)
        preds[no_coverage] = ABSTAIN
        return preds

    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Normalised weighted vote counts as probability estimates."""
        weighted = self._weighted_counts(label_matrix)
        totals = weighted.sum(axis=1, keepdims=True)
        no_coverage = (totals == 0.0).ravel()
        totals[no_coverage] = 1.0
        proba = weighted / totals
        proba[no_coverage] = 1.0 / self.num_classes_
        return proba

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _weighted_counts(self, label_matrix: np.ndarray) -> np.ndarray:
        """Build a (n_samples, num_classes) matrix of weighted vote counts."""
        n_samples, n_lfs = label_matrix.shape
        counts = np.zeros((n_samples, self.num_classes_), dtype=np.float64)
        for c in range(self.num_classes_):
            # (n_samples, n_lfs) indicator that LF voted for class c
            indicator = (label_matrix == c).astype(np.float64)
            # weight each LF and sum
            counts[:, c] = indicator @ self.weights_
        return counts

    @staticmethod
    def _majority_of(sub_matrix: np.ndarray, num_classes: int) -> np.ndarray:
        """Compute the majority vote across columns for each row.

        Returns -1 for rows where every entry is ABSTAIN.
        """
        n, _ = sub_matrix.shape
        counts = np.zeros((n, num_classes), dtype=np.int64)
        for c in range(num_classes):
            counts[:, c] = (sub_matrix == c).sum(axis=1)
        totals = counts.sum(axis=1)
        majority = counts.argmax(axis=1).astype(np.intp)
        majority[totals == 0] = ABSTAIN
        return majority
