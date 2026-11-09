"""Whole-pipeline (nested) evaluation used for the study's pipeline-inflation estimates.

Three schemes, each repeated over seeds with 5 outer folds and 4 inner folds:
  leaky_pipeline             random outer folds, random inner folds for tuning C
  grouped_pipeline           subject-grouped outer and inner folds
  grouped_eval_leaky_tuning  subject-grouped outer folds, random inner folds
Pipeline inflation = mean(leaky_pipeline) - mean(grouped_pipeline).
Preprocessing (prevalence filter, CLR, scaling) is refitted inside every training fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from analysis.preprocessing import classifier

C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
SCHEMES = ("leaky_pipeline", "grouped_pipeline", "grouped_eval_leaky_tuning")


def _grouped_splits(n_folds, seed, X, y, g):
    out = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(X, y, groups=g))
    for train, test in out:
        if set(g[train]) & set(g[test]):
            raise AssertionError("subject overlap in grouped split")
    return out


def nested_auc(X, y, subjects, seeds=range(5), c_grid=C_GRID):
    """Return one row per seed and scheme with the outer-fold-averaged ROC AUC."""
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(subjects)
    rows = []
    for seed in seeds:
        for scheme in SCHEMES:
            if scheme == "leaky_pipeline":
                outer = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
            else:
                outer = _grouped_splits(5, seed, X, y, g)
            aucs = []
            for train, test in outer:
                if scheme == "grouped_pipeline":
                    inner = _grouped_splits(4, seed, X[train], y[train], g[train])
                else:
                    inner = StratifiedKFold(4, shuffle=True, random_state=seed)
                search = GridSearchCV(classifier(1.0), {"logisticregression__C": list(c_grid)},
                                      cv=inner, scoring="roc_auc", n_jobs=1).fit(X[train], y[train])
                aucs.append(roc_auc_score(y[test], search.predict_proba(X[test])[:, 1]))
            rows.append({"seed": seed, "scheme": scheme, "auc": float(np.mean(aucs))})
    return pd.DataFrame(rows)


def summarize(table):
    """Mean AUC per scheme, pipeline inflation and the loss from leaky tuning."""
    m = table.groupby("scheme").auc.mean()
    return {"leaky_pipeline": float(m["leaky_pipeline"]),
            "grouped_pipeline": float(m["grouped_pipeline"]),
            "grouped_eval_leaky_tuning": float(m["grouped_eval_leaky_tuning"]),
            "pipeline_inflation": float(m["leaky_pipeline"] - m["grouped_pipeline"]),
            "leaky_tuning_loss": float(m["grouped_pipeline"] - m["grouped_eval_leaky_tuning"])}
