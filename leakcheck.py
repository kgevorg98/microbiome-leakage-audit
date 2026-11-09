"""Diagnostics for subject-level leakage in cross-validated classifiers.

Compare sample-wise and subject-grouped cross-validation of the same model.
All estimator preprocessing must be fitted inside the estimator pipeline.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score

__version__ = "0.3.1"
__all__ = ["audit", "label_constancy", "sibling_dose", "design_dose",
           "feature_icc", "predict_inflation", "correct_auc", "BETA", "BETA_CI"]

BETA = 0.594
BETA_CI = (0.398, 1.195)


def design_dose(samples_per_subject: float, n_folds: int) -> float:
    d, k = float(samples_per_subject), int(n_folds)
    if d < 1 or k < 2:
        raise ValueError("need samples_per_subject >= 1 and n_folds >= 2")
    return (d - 1.0) * (k - 1.0) / k


def sibling_dose(subjects, folds=None, n_folds: int | None = None) -> float:
    s = pd.Series(np.asarray(subjects)).reset_index(drop=True)
    if folds is None:
        if n_folds is None:
            raise ValueError("give either folds or n_folds")
        n_i = s.map(s.value_counts()).to_numpy(dtype=float)
        return float(np.mean((n_i - 1.0) * (n_folds - 1.0) / n_folds))
    f = pd.Series(np.asarray(folds)).reset_index(drop=True)
    pair = pd.Series(list(zip(s, f)))
    return float(np.mean(s.map(s.value_counts()).to_numpy()
                         - pair.map(pair.value_counts()).to_numpy()))


def label_constancy(y, subjects) -> float:
    """Fraction of subjects with changing labels across observations."""
    n = pd.Series(np.asarray(y)).groupby(np.asarray(subjects)).nunique()
    return float((n > 1).mean())


def _icc(v, g, min_groups=8):
    df = pd.DataFrame({"v": v, "g": g})
    k = df.groupby("g").size()
    if (k >= 2).sum() < min_groups:
        return np.nan
    df = df[df.g.isin(k[k >= 2].index)]
    gm = df.groupby("g").v.mean()
    n = df.groupby("g").size()
    grand = df.v.mean()
    msb = (n * (gm - grand) ** 2).sum() / (len(gm) - 1)
    msw = df.groupby("g").v.apply(
        lambda s: ((s - s.mean()) ** 2).sum()).sum() / max(len(df) - len(gm), 1)
    n0 = (len(df) - (n ** 2).sum() / len(df)) / (len(gm) - 1)
    vb = (msb - msw) / n0
    return float(np.clip(vb / (vb + msw), 0, 1)) if (vb + msw) > 0 else np.nan


def feature_icc(X, y, subjects) -> float:
    """Median within-label one-way ICC across features."""
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(subjects)
    per = [[_icc(X[y == lab, j], g[y == lab])
            for j in range(X.shape[1])] for lab in np.unique(y)]
    return float(np.nanmedian(np.nanmean(np.array(per, dtype=float), axis=0)))


def predict_inflation(
    icc: float, m: float, auc_grouped: float, beta: float = BETA
) -> float:
    if not 0.0 <= icc < 1.0:
        raise ValueError("ICC must be in [0, 1)")
    r = icc / (1.0 - icc)
    return float((1.0 - auc_grouped) * (1.0 - np.exp(-beta * r * m)))


def correct_auc(
    auc_random: float, icc: float, m: float, beta: float = BETA,
    allow_unvalidated: bool = False
) -> float:
    if not allow_unvalidated:
        raise RuntimeError(
            "correct_auc is not validated. Measure grouped AUC with audit() "
            "or pass allow_unvalidated=True for exploratory use."
        )
    warnings.warn("correct_auc is exploratory and not validated", RuntimeWarning)
    f = 1.0 - np.exp(-beta * (icc / (1.0 - icc)) * m)
    if f > 0.95:
        return float("nan")
    return float((auc_random - f) / (1.0 - f))


def audit(
    estimator_factory, X, y, subjects, n_folds: int = 5, seeds=range(10)
) -> dict:
    """Compare mean fold-averaged ROC AUC using random and grouped splits."""
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(subjects)
    rnd = [
        cross_val_score(
            estimator_factory(), X, y, scoring="roc_auc", n_jobs=1,
            cv=StratifiedKFold(
                n_folds, shuffle=True, random_state=s)).mean()
        for s in seeds
    ]
    grp = [
        cross_val_score(
            estimator_factory(), X, y, groups=g, scoring="roc_auc", n_jobs=1,
            cv=StratifiedGroupKFold(
                n_folds, shuffle=True, random_state=s)).mean()
        for s in seeds
    ]
    rnd, grp = np.array(rnd), np.array(grp)
    counts = pd.Series(g).value_counts()
    return {
        "n_samples": int(len(y)),
        "n_subjects": int(counts.size),
        "samples_per_subject": float(len(y) / counts.size),
        "sibling_dose": sibling_dose(g, n_folds=n_folds),
        "label_varies_fraction": label_constancy(y, g),
        "auc_random": float(rnd.mean()),
        "auc_grouped": float(grp.mean()),
        "inflation": float((rnd - grp).mean()),
        "inflation_sd": float((rnd - grp).std(ddof=1))
        if len(rnd) > 1 else float("nan"),
    }
