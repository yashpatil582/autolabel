"""Per-labeling-function analysis of a label matrix."""

from __future__ import annotations

import numpy as np

# Sentinel: -1 means the LF abstained on this example.
ABSTAIN = -1


class LFAnalysis:
    """Analyse coverage, accuracy, overlap, and conflicts of labeling functions.

    Parameters
    ----------
    label_matrix : np.ndarray
        Integer-encoded matrix of shape ``(n_examples, n_lfs)``.
        ``-1`` denotes abstain.
    labels : list[str]
        Ground-truth string labels, one per example (same length as
        ``label_matrix.shape[0]``).
    label_space : list[str]
        The ordered list of all possible class labels.  Index ``i`` in
        ``label_space`` corresponds to integer ``i`` in the label matrix.
    """

    def __init__(
        self,
        label_matrix: np.ndarray,
        labels: list[str],
        label_space: list[str],
    ) -> None:
        self.label_matrix = label_matrix
        self.labels = labels
        self.label_space = label_space

        self._n_examples, self._n_lfs = label_matrix.shape

        # Encode ground-truth labels to integers for fast comparison
        self._label_to_int = {lbl: i for i, lbl in enumerate(label_space)}
        self._encoded_labels = np.array([self._label_to_int[lbl] for lbl in labels], dtype=int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def per_lf_stats(self) -> list[dict]:
        """Compute per-LF statistics.

        Returns a list of dicts (one per LF, in column order) with keys:

        * ``lf_index`` -- column index in the label matrix
        * ``coverage`` -- fraction of examples the LF labels (non-abstain)
        * ``accuracy`` -- fraction of labelled examples where the LF is correct
        * ``overlap`` -- fraction of examples where this LF *and* at least one
          other LF both voted (non-abstain)
        * ``conflict`` -- fraction of examples where this LF disagrees with at
          least one other non-abstain LF
        """
        stats: list[dict] = []

        for j in range(self._n_lfs):
            col = self.label_matrix[:, j]
            labelled_mask = col != ABSTAIN

            # Coverage
            n_labelled = int(labelled_mask.sum())
            coverage = n_labelled / self._n_examples if self._n_examples else 0.0

            # Accuracy (among labelled)
            if n_labelled > 0:
                correct = int((col[labelled_mask] == self._encoded_labels[labelled_mask]).sum())
                accuracy = correct / n_labelled
            else:
                accuracy = 0.0

            # Overlap: this LF voted AND at least one other LF also voted
            other_voted = np.zeros(self._n_examples, dtype=bool)
            for k in range(self._n_lfs):
                if k != j:
                    other_voted |= self.label_matrix[:, k] != ABSTAIN
            overlap_count = int((labelled_mask & other_voted).sum())
            overlap = overlap_count / self._n_examples if self._n_examples else 0.0

            # Conflict: this LF voted AND at least one other LF voted differently
            conflict_count = 0
            if n_labelled > 0:
                for i in np.where(labelled_mask)[0]:
                    this_vote = col[i]
                    for k in range(self._n_lfs):
                        if k != j:
                            other_vote = self.label_matrix[i, k]
                            if other_vote != ABSTAIN and other_vote != this_vote:
                                conflict_count += 1
                                break  # one conflict is enough for this example
            conflict = conflict_count / self._n_examples if self._n_examples else 0.0

            stats.append(
                {
                    "lf_index": j,
                    "coverage": coverage,
                    "accuracy": accuracy,
                    "overlap": overlap,
                    "conflict": conflict,
                }
            )

        return stats

    def summary_table(self) -> str:
        """Return a human-readable formatted table of per-LF statistics."""
        stats = self.per_lf_stats()

        header = (
            f"{'LF':>5s}  {'Coverage':>9s}  {'Accuracy':>9s}  {'Overlap':>9s}  {'Conflict':>9s}"
        )
        separator = "-" * len(header)
        lines = [header, separator]

        for s in stats:
            lines.append(
                f"{s['lf_index']:5d}  "
                f"{s['coverage']:9.4f}  "
                f"{s['accuracy']:9.4f}  "
                f"{s['overlap']:9.4f}  "
                f"{s['conflict']:9.4f}"
            )

        # Averages row
        if stats:
            avg_cov = np.mean([s["coverage"] for s in stats])
            avg_acc = np.mean([s["accuracy"] for s in stats])
            avg_ovl = np.mean([s["overlap"] for s in stats])
            avg_con = np.mean([s["conflict"] for s in stats])
            lines.append(separator)
            lines.append(
                f"{'Mean':>5s}  {avg_cov:9.4f}  {avg_acc:9.4f}  {avg_ovl:9.4f}  {avg_con:9.4f}"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LFAnalysis(n_examples={self._n_examples}, n_lfs={self._n_lfs}, "
            f"num_classes={len(self.label_space)})"
        )
