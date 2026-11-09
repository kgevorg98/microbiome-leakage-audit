import sys
import types
import numpy as np
import pandas as pd
from analysis.curated_data import load_cohort


def test_loader_filters_and_deduplicates(tmp_path, monkeypatch):
    root = tmp_path
    (root / "curation").mkdir()
    (root / "matrices").mkdir()
    metadata = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e", "f"],
        "subject_id": ["p1", "p1", "p2", "p2", "p3", "p4"],
        "body_site": ["stool"] * 5 + ["skin"],
        "disease": ["healthy", "healthy", "Crohn", "Crohn", "unknown", "Crohn"],
        "sex": ["female", "female", "male", "male", "female", "female"],
    })
    metadata.to_csv(root / "curation" / "Mock.tsv", sep="\t", index=False)
    (root / "matrices" / "Mock.relative_abundance.rda").touch()
    abundance = pd.DataFrame({
        "a": [25., 75.], "b": [25., 75.], "c": [70., 30.],
        "d": [60., 40.], "e": [30., 70.], "f": [60., 40.],
    }, index=["k__Bacteria|s__Species_A", "k__Bacteria|s__Species_B"])
    monkeypatch.setitem(sys.modules, "pyreadr", types.SimpleNamespace(
        read_r=lambda path: {"a": abundance}))
    X, y, g, names = load_cohort(root, "Mock", "disease")
    assert X.shape == (3, 2)
    assert y.tolist() == [0, 1, 1]
    assert g.tolist() == ["p1", "p2", "p2"]
    assert len(names) == 2


def _write_mock(root, monkeypatch, cohort, metadata, abundance):
    (root / "curation").mkdir(exist_ok=True)
    (root / "matrices").mkdir(exist_ok=True)
    metadata.to_csv(root / "curation" / f"{cohort}.tsv", sep="\t", index=False)
    (root / "matrices" / f"{cohort}.relative_abundance.rda").touch()
    monkeypatch.setitem(sys.modules, "pyreadr", types.SimpleNamespace(read_r=lambda path: {"a": abundance}))


def _profiles(n, rng, species=8):
    v = rng.dirichlet(np.ones(species), size=n) * 100
    return pd.DataFrame(v.T, index=[f"k__Bacteria|s__Sp{i}" for i in range(species)])


def test_loader_removes_profiles_shared_by_different_subjects(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    ab = _profiles(6, rng)
    ab.columns = list("abcdef")
    ab["f"] = ab["c"]  # identical profile recorded under a different subject
    md = pd.DataFrame({"sample_id": list("abcdef"), "subject_id": ["p1", "p1", "p2", "p3", "p3", "p4"],
                       "body_site": ["stool"] * 6, "disease": ["healthy", "healthy", "Crohn", "Crohn", "Crohn", "healthy"],
                       "sex": ["female"] * 6})
    _write_mock(tmp_path, monkeypatch, "Mock", md, ab)
    X, y, g, names, log = load_cohort(tmp_path, "Mock", "disease", return_log=True)
    assert g.tolist() == ["p1", "p1", "p3", "p3"]
    assert log["dropped_cross_subject_identical"] == 2
    assert log["n_after_clean"] == 4


def test_yassour_sex_keeps_all_samples_of_children(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    ids = ["m1", "c1a", "c1b", "c1c", "m2", "c2a", "c2b"]
    ab = _profiles(len(ids), rng)
    ab.columns = ids
    md = pd.DataFrame({"sample_id": ids, "subject_id": ["M1", "C1", "C1", "C1", "M2", "C2", "C2"],
                       "body_site": ["stool"] * 7, "disease": ["healthy"] * 7,
                       "sex": ["female", "male", "male", "male", "female", "female", "female"],
                       "age_group": ["Adult", "Newborn", "Infant", "Infant", "Adult", "Newborn", "Infant"]})
    _write_mock(tmp_path, monkeypatch, "YassourM_2018", md, ab)
    X, y, g, names = load_cohort(tmp_path, "YassourM_2018", "sex")
    assert g.tolist() == ["C1", "C1", "C1", "C2", "C2"]
    assert y.tolist() == [1, 1, 1, 0, 0]
