"""Tests for label model implementations."""

import numpy as np
import pytest

from autolabel.label_model.majority_vote import MajorityVoteLabelModel
from autolabel.label_model.weighted_vote import WeightedVoteLabelModel
from autolabel.label_model.generative import GenerativeLabelModel


class TestMajorityVote:
    def test_unanimous(self, sample_label_matrix):
        model = MajorityVoteLabelModel()
        model.fit(sample_label_matrix, num_classes=3)
        preds = model.predict(sample_label_matrix)
        assert preds[2] == 2  # unanimous class 2

    def test_no_coverage(self, sample_label_matrix):
        model = MajorityVoteLabelModel()
        model.fit(sample_label_matrix, num_classes=3)
        preds = model.predict(sample_label_matrix)
        assert preds[4] == -1  # all abstain

    def test_majority_wins(self, sample_label_matrix):
        model = MajorityVoteLabelModel()
        model.fit(sample_label_matrix, num_classes=3)
        preds = model.predict(sample_label_matrix)
        assert preds[0] == 0  # two votes for class 0
        assert preds[1] == 1  # two votes for class 1

    def test_predict_proba_shape(self, sample_label_matrix):
        model = MajorityVoteLabelModel()
        model.fit(sample_label_matrix, num_classes=3)
        proba = model.predict_proba(sample_label_matrix)
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


class TestWeightedVote:
    def test_predictions(self, sample_label_matrix):
        model = WeightedVoteLabelModel()
        model.fit(sample_label_matrix, num_classes=3)
        preds = model.predict(sample_label_matrix)
        assert preds[2] == 2  # unanimous
        assert preds[4] == -1  # no coverage

    def test_weights_in_range(self, sample_label_matrix):
        model = WeightedVoteLabelModel()
        model.fit(sample_label_matrix, num_classes=3)
        assert all(0.5 <= w <= 1.0 for w in model.weights_)


class TestGenerativeModel:
    def test_fit_predict(self, sample_label_matrix):
        model = GenerativeLabelModel(n_epochs=50, seed=42)
        model.fit(sample_label_matrix, num_classes=3)
        preds = model.predict(sample_label_matrix)
        assert preds.shape == (5,)
        assert preds[4] == -1  # no coverage

    def test_predict_proba_sums_to_one(self, sample_label_matrix):
        model = GenerativeLabelModel(n_epochs=50, seed=42)
        model.fit(sample_label_matrix, num_classes=3)
        proba = model.predict_proba(sample_label_matrix)
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_converges_on_clear_signal(self):
        """When LFs agree perfectly, model should learn high accuracies."""
        # 100 samples, 3 LFs, 2 classes: all agree on class 0 for first 50, class 1 for rest
        matrix = np.zeros((100, 3), dtype=int)
        matrix[50:] = 1
        model = GenerativeLabelModel(n_epochs=200, seed=42)
        model.fit(matrix, num_classes=2)
        preds = model.predict(matrix)
        assert (preds[:50] == 0).all()
        assert (preds[50:] == 1).all()

    def test_not_fitted_raises(self):
        model = GenerativeLabelModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([[0, 1]]))
