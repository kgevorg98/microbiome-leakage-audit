import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from analysis.preprocessing import PrevalenceCLR, classifier
from leakcheck import audit
from analysis.nested import nested_auc, summarize, SCHEMES


def test_prevalence_learned_on_training_only():
    a = np.array([[1., 0., 1.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])
    p = PrevalenceCLR(prevalence=.5).fit(a[:2])
    assert p.keep_.tolist() == [True, False, True]
    assert p.transform(a[2:]).shape == (2, 2)


def test_preprocessing_rejects_invalid_matrix():
    with pytest.raises(ValueError):
        PrevalenceCLR().fit(np.array([[0., -1.]]))
    with pytest.raises(ValueError):
        PrevalenceCLR(prevalence=1).fit(np.array([[0., 0.]]))


def test_grouped_cv_has_disjoint_people():
    g = np.repeat(np.arange(30), 3)
    y = np.repeat(np.arange(30) % 2, 3)
    X = np.ones((len(y), 5))
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        assert not set(g[train]).intersection(g[test])


def test_pipeline_audit_on_repeated_samples():
    rng = np.random.default_rng(21)
    g = np.repeat(np.arange(40), 4)
    y = np.repeat(np.arange(40) % 2, 4)
    X = np.exp(rng.normal(0, .6, (40, 30))[g] + rng.normal(0, .2, (len(g), 30)))
    out = audit(lambda: classifier(.1), X, y, g, n_folds=4, seeds=range(2))
    assert 0 <= out["auc_grouped"] <= 1
    assert out["n_subjects"] == 40


def test_nested_pipeline_schemes_and_inflation():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(24), 4)
    y = np.repeat(np.arange(24) % 2, 4)
    X = np.exp(rng.normal(0, .8, (24, 15))[g] + rng.normal(0, .15, (len(g), 15)))
    table = nested_auc(X, y, g, seeds=range(1), c_grid=[0.01, 1.0])
    assert sorted(table.scheme) == sorted(SCHEMES)
    out = summarize(table)
    assert out["pipeline_inflation"] == out["leaky_pipeline"] - out["grouped_pipeline"]
    assert out["pipeline_inflation"] > 0.1  # labels carry no signal beyond subject identity
