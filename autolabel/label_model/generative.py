"""EM-based generative label model (Snorkel-style, from scratch)."""

from __future__ import annotations

import numpy as np

from autolabel.label_model.base import BaseLabelModel

ABSTAIN = -1
_EPS = 1e-10


class GenerativeLabelModel(BaseLabelModel):
    """Expectation-Maximisation label model inspired by Snorkel's data programming.

    Models P(LF_j = l | Y = y) for each labeling function *j*, true label *y*,
    and observed vote *l*.  The E-step computes posterior P(Y | LF votes) and
    the M-step updates the LF accuracy parameters from expected sufficient
    statistics.

    Everything is vectorised with NumPy — no Python loops over data points.
    """

    def __init__(self, n_epochs: int = 500, seed: int = 42) -> None:
        self.n_epochs = n_epochs
        self.seed = seed
        # Filled after fit()
        self.num_classes_: int = 0
        self.n_lfs_: int = 0
        self.mu_: np.ndarray | None = None  # (n_lfs, C, C)
        self.class_prior_: np.ndarray | None = None  # (C,)

    def fit(self, label_matrix: np.ndarray, num_classes: int) -> GenerativeLabelModel:
        """Run EM to learn LF accuracy parameters and class prior."""
        rng = np.random.RandomState(self.seed)
        n_samples, n_lfs = label_matrix.shape
        C = num_classes
        self.num_classes_ = C
        self.n_lfs_ = n_lfs

        # --- initialisation ---------------------------------------------------
        # class_prior: uniform
        class_prior = np.full(C, 1.0 / C)

        # mu[j, y, l] = P(LF_j = l | Y = y)
        # Initialise with slight diagonal dominance
        mu = np.full((n_lfs, C, C), 1.0 / C)
        for j in range(n_lfs):
            for y in range(C):
                mu[j, y, y] += 0.3
            # renormalise each row
            mu[j] = mu[j] / mu[j].sum(axis=1, keepdims=True)

        # Add tiny noise for symmetry breaking
        mu += rng.uniform(0, 0.01, mu.shape)
        mu = mu / mu.sum(axis=2, keepdims=True)
        mu = np.clip(mu, _EPS, 1.0 - _EPS)

        # Pre-compute coverage masks: (n_samples, n_lfs), True where LF votes
        coverage = label_matrix != ABSTAIN  # (n, m)

        # --- EM loop ----------------------------------------------------------
        for epoch in range(self.n_epochs):
            # === E-step: compute q[i, y] = P(Y=y | LF votes for sample i) ===
            # log P(Y=y) + sum_j log P(LF_j=l_ij | Y=y)  [only for active LFs]
            log_prior = np.log(class_prior + _EPS)  # (C,)

            # Build log-likelihood contributions: (n_samples, C)
            log_lik = np.tile(log_prior, (n_samples, 1))  # start with prior

            for j in range(n_lfs):
                active = coverage[:, j]  # (n_samples,) bool
                if not active.any():
                    continue
                active_idx = np.where(active)[0]
                votes_j = label_matrix[active_idx, j]  # (n_active,) class indices
                # mu[j, :, votes_j] -> (C, n_active), transpose -> (n_active, C)
                lf_probs = mu[j][:, votes_j].T  # (n_active, C)
                log_lik[active_idx] += np.log(lf_probs + _EPS)

            # Normalise in log space
            log_lik_max = log_lik.max(axis=1, keepdims=True)
            log_lik -= log_lik_max
            q = np.exp(log_lik)
            q = q / (q.sum(axis=1, keepdims=True) + _EPS)  # (n_samples, C)

            # === M-step: update mu and class_prior ===
            # Update class prior
            class_prior = q.mean(axis=0)
            class_prior = np.clip(class_prior, _EPS, 1.0)
            class_prior = class_prior / class_prior.sum()

            # Update mu[j, y, l]
            for j in range(n_lfs):
                active = coverage[:, j]
                n_active = active.sum()
                if n_active == 0:
                    continue
                votes_j = label_matrix[active, j]  # (n_active,)
                q_active = q[active]  # (n_active, C)

                # For each (y, cls) pair, accumulate:
                # mu_new[y, cls] = sum_i q[i,y] * 1(vote_i == cls) / sum_i q[i,y]
                new_mu_j = np.zeros((C, C))
                for cls in range(C):
                    indicator = (votes_j == cls).astype(np.float64)  # (n_active,)
                    # sum_i q[i,y] * indicator[i] for each y
                    new_mu_j[:, cls] = (q_active * indicator[:, None]).sum(axis=0)

                # Normalise: each row y sums to 1
                row_sums = new_mu_j.sum(axis=1, keepdims=True) + _EPS
                new_mu_j = new_mu_j / row_sums
                mu[j] = np.clip(new_mu_j, _EPS, 1.0 - _EPS)

        self.mu_ = mu
        self.class_prior_ = class_prior
        return self

    def predict(self, label_matrix: np.ndarray) -> np.ndarray:
        """Predict labels using learned parameters."""
        proba = self.predict_proba(label_matrix)
        coverage = (label_matrix != ABSTAIN).any(axis=1)
        preds = proba.argmax(axis=1).astype(np.intp)
        preds[~coverage] = ABSTAIN
        return preds

    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Compute posterior P(Y | LF votes) using learned parameters."""
        if self.mu_ is None or self.class_prior_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples, n_lfs = label_matrix.shape
        C = self.num_classes_
        coverage = label_matrix != ABSTAIN

        log_prior = np.log(self.class_prior_ + _EPS)
        log_lik = np.tile(log_prior, (n_samples, 1))

        for j in range(n_lfs):
            active = coverage[:, j]
            if not active.any():
                continue
            active_idx = np.where(active)[0]
            votes_j = label_matrix[active_idx, j]
            lf_probs = self.mu_[j][:, votes_j].T  # (n_active, C)
            log_lik[active_idx] += np.log(lf_probs + _EPS)

        log_lik_max = log_lik.max(axis=1, keepdims=True)
        log_lik -= log_lik_max
        proba = np.exp(log_lik)
        proba = proba / (proba.sum(axis=1, keepdims=True) + _EPS)

        # No-coverage samples get uniform
        no_cov = ~coverage.any(axis=1)
        proba[no_cov] = 1.0 / C

        return proba
