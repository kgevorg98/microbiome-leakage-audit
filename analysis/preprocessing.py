"""Leakage-safe preprocessing for species-level abundance profiles."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class PrevalenceCLR(BaseEstimator, TransformerMixin):
    """Learn species prevalence on training samples; transform with CLR.

    Input values are nonnegative relative abundances, consistently scaled
    across training and evaluation samples (percentage units in the study).
    """

    def __init__(self, prevalence=0.1, pseudocount=1e-5):
        self.prevalence = prevalence
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
            raise ValueError("Expected nonempty samples-by-features matrix")
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Abundances must be finite and nonnegative")
        if not 0 <= self.prevalence <= 1 or self.pseudocount <= 0:
            raise ValueError("Invalid preprocessing parameters")
        self.n_features_in_ = a.shape[1]
        self.keep_ = (a > 0).mean(axis=0) >= self.prevalence
        if not self.keep_.any():
            raise ValueError("No features satisfy the training-fold prevalence filter")
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "keep_")
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted training data")
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Abundances must be finite and nonnegative")
        log = np.log(a[:, self.keep_] + self.pseudocount)
        return log - log.mean(axis=1, keepdims=True)


def classifier(C=0.1):
    return make_pipeline(
        PrevalenceCLR(), StandardScaler(),
        LogisticRegression(C=C, max_iter=5000),
    )
