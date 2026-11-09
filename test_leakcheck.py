"""Unit tests for the standalone leakcheck module."""
import math
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import leakcheck as lc


def _toy(n_sub=40, d=4, icc=0.7, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_sub), d)
    y = np.repeat(np.arange(n_sub) % 2, d)
    b = rng.normal(0, np.sqrt(icc), (n_sub, 20))[g]
    e = rng.normal(0, np.sqrt(1 - icc), (len(g), 20))
    X = b + e
    X[:, :3] += 0.3 * y[:, None]
    return X, y, g


def test_design_dose():
    assert lc.design_dose(1, 5) == 0.0
    assert math.isclose(lc.design_dose(4, 5), 3 * 4 / 5)


def test_sibling_dose_analytic_matches_design():
    g = np.repeat(np.arange(30), 3)
    assert math.isclose(lc.sibling_dose(g, n_folds=5), lc.design_dose(3, 5))


def test_sibling_dose_grouped_folds_is_zero():
    X, y, g = _toy()
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5).split(X, y, g)):
        folds[te] = k
    assert lc.sibling_dose(g, folds=folds) == 0.0


def test_sibling_dose_random_folds_near_analytic():
    X, y, g = _toy(n_sub=200, d=4)
    vals = []
    for s in range(20):
        folds = np.empty(len(y), int)
        for k, (_, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)):
            folds[te] = k
        vals.append(lc.sibling_dose(g, folds=folds))
    assert abs(np.mean(vals) - lc.design_dose(4, 5)) < 0.05


def test_label_constancy():
    g = np.array([0, 0, 1, 1, 2, 2])
    assert lc.label_constancy([0, 0, 1, 1, 0, 0], g) == 0.0
    assert math.isclose(lc.label_constancy([0, 1, 1, 1, 0, 0], g), 1 / 3)


def test_feature_icc_recovers_level():
    X, y, g = _toy(n_sub=80, d=4, icc=0.6, seed=1)
    assert abs(lc.feature_icc(X, y, g) - 0.6) < 0.1


def test_predict_inflation_zeros_and_bound():
    assert lc.predict_inflation(0.0, 3.0, 0.7) == 0.0
    assert lc.predict_inflation(0.6, 0.0, 0.7) == 0.0
    assert lc.predict_inflation(0.99, 100.0, 0.7) <= 0.3 + 1e-12


def test_correct_auc_guarded():
    try:
        lc.correct_auc(0.9, 0.5, 2.0)
        raise AssertionError("should raise")
    except RuntimeError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(lc.correct_auc(
            0.99, 0.95, 20.0, allow_unvalidated=True))
        a = 0.7
        f = 1 - math.exp(-lc.BETA * 1.0 * 1.0)
        r = a + (1 - a) * f
        assert math.isclose(
            lc.correct_auc(r, 0.5, 1.0, allow_unvalidated=True), a)


def test_audit_detects_inflation():
    X, y, g = _toy(n_sub=40, d=4, icc=0.8, seed=2)
    out = lc.audit(
        lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        X, y, g, n_folds=4, seeds=range(3))
    assert out["label_varies_fraction"] == 0.0
    assert out["inflation"] > 0.1


def test_audit_input_validation():
    X, y, g = _toy(n_sub=20, d=2)
    factory = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    for bad in [dict(n_folds=1), dict(seeds=[]), dict(y=np.zeros(len(y))), dict(subjects=g[:-1])]:
        kw = dict(X=X, y=y, subjects=g, n_folds=2, seeds=range(1))
        kw.update(bad)
        try:
            lc.audit(factory, **kw)
            raise AssertionError("Expected a validation failure")
        except ValueError:
            pass
