"""Scoring metrics for AutoLabel predictions and label matrices."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Sentinel value used when a labeling function abstains.
ABSTAIN = -1


def compute_f1(
    y_true: list[str],
    y_pred: list[str | None],
    label_space: list[str],
    penalize_abstains: bool = True,
) -> float:
    """Compute micro-averaged F1 between ground-truth and predicted labels.

    When *penalize_abstains* is True (default), abstaining predictions are
    treated as incorrect — this prevents inflated scores at low coverage.
    When False, only non-abstaining predictions are scored.
    """
    le = LabelEncoder()
    le.fit(label_space + ["__WRONG__"])
    valid_labels = set(label_space)

    true_encoded: list[int] = []
    pred_encoded: list[int] = []

    wrong_idx = le.transform(["__WRONG__"])[0]

    for yt, yp in zip(y_true, y_pred):
        is_abstain = yp is None or yp == "ABSTAIN"

        if is_abstain and not penalize_abstains:
            continue

        true_encoded.append(le.transform([yt])[0])
        if is_abstain:
            pred_encoded.append(wrong_idx)
        elif yp in valid_labels:
            pred_encoded.append(le.transform([yp])[0])
        else:
            pred_encoded.append(wrong_idx)

    if not true_encoded:
        return 0.0

    return float(
        f1_score(
            np.array(true_encoded),
            np.array(pred_encoded),
            average="micro",
            zero_division=0,
        )
    )


def compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Compute accuracy, ignoring None / ABSTAIN predictions.

    Returns 0.0 when there are no valid predictions.
    """
    filtered_true: list[str] = []
    filtered_pred: list[str] = []
    for yt, yp in zip(y_true, y_pred):
        if yp is not None and yp != "ABSTAIN":
            filtered_true.append(yt)
            filtered_pred.append(yp)

    if not filtered_true:
        return 0.0

    return float(accuracy_score(filtered_true, filtered_pred))


def compute_coverage(predictions: list[str | None]) -> float:
    """Fraction of predictions that are not None and not ``'ABSTAIN'``.

    A coverage of 1.0 means every example received a concrete label.
    """
    if not predictions:
        return 0.0
    covered = sum(1 for p in predictions if p is not None and p != "ABSTAIN")
    return covered / len(predictions)


def compute_conflict_rate(label_matrix: np.ndarray) -> float:
    """Fraction of data-points that have conflicting non-abstain LF votes.

    Parameters
    ----------
    label_matrix : np.ndarray
        Shape ``(n_examples, n_lfs)`` with integer-encoded labels.
        ``-1`` encodes abstain.

    Returns
    -------
    float
        The fraction of rows where at least two LFs voted for different
        (non-abstain) labels.
    """
    n = label_matrix.shape[0]
    if n == 0:
        return 0.0

    conflicts = 0
    for i in range(n):
        row = label_matrix[i]
        non_abstain = row[row != ABSTAIN]
        if len(non_abstain) >= 2 and len(set(non_abstain.tolist())) > 1:
            conflicts += 1

    return conflicts / n


def per_class_f1(
    y_true: list[str],
    y_pred: list[str | None],
    label_space: list[str],
) -> dict[str, float]:
    """Per-class F1 scores for each label in ``label_space``.

    Abstains are treated as wrong predictions so coverage is penalised.
    """
    le = LabelEncoder()
    le.fit(label_space + ["__WRONG__"])
    valid_labels = set(label_space)
    wrong_idx = le.transform(["__WRONG__"])[0]

    encoded_true = []
    encoded_pred = []
    for yt, yp in zip(y_true, y_pred):
        encoded_true.append(le.transform([yt])[0])
        if yp is None or yp == "ABSTAIN":
            encoded_pred.append(wrong_idx)
        elif yp in valid_labels:
            encoded_pred.append(le.transform([yp])[0])
        else:
            encoded_pred.append(wrong_idx)

    if not encoded_true:
        return {label: 0.0 for label in label_space}

    label_indices = le.transform(label_space)
    scores = f1_score(
        np.array(encoded_true),
        np.array(encoded_pred),
        labels=label_indices,
        average=None,
        zero_division=0,
    )

    return {label: float(score) for label, score in zip(label_space, scores)}
