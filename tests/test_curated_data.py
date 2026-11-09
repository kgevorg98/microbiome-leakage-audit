import sys
import types
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
