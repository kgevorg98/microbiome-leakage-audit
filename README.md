# leakcheck

`leakcheck` evaluates subject-level leakage in classifiers trained on repeated microbiome observations. Its principal diagnostic is the difference between sample-wise and subject-grouped cross-validation, accompanied by descriptive measures of within-subject replication.

This repository contains the standalone diagnostic, tests, a synthetic-data notebook, and a small workflow for analyzing locally available species-level abundance profiles. It does **not** contain manuscript files, participant-level data, or the complete computational pipeline used to generate all published tables and figures.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
```

Install `requirements-analysis.txt` as well when running analyses using exported `.rda` matrices.

## Standalone API

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from leakcheck import audit

result = audit(
    estimator_factory=lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000),
    ),
    X=features,
    y=labels,
    subjects=subject_ids,
    n_folds=5,
)
print(result)
```

The estimator factory must construct a new, unfitted estimator on each call. Any learned feature filtering, transformation, imputation, scaling, and tuning should be confined to training folds. Group identifiers must distinguish individuals, not samples.

The random-versus-grouped AUC difference measures sensitivity to the validation scheme. It is **not** a universal correction for a published AUC. The exploratory inverse function `correct_auc` is disabled by default because it has not generalized reliably.

## Gut-metagenome example

For a cohort exported from curatedMetagenomicData, arrange local files as:

```text
local-data/
  curation/
    HMP_2019_ibdmdb.tsv
  matrices/
    HMP_2019_ibdmdb.relative_abundance.rda
```

Then execute:

```bash
python -m pip install -r requirements-analysis.txt
python -m analysis.run_cohort --data-dir local-data --cohort HMP_2019_ibdmdb --label disease --folds 5 --seeds 10 --C 0.1
```

The `analysis.preprocessing.PrevalenceCLR` transformer learns the feature prevalence filter on the training fold and then applies the CLR transformation to training and test samples consistently. The loader implements the necessary stool/sample filtering and a duplicate check. It is a **convenience loader**, not an audited replacement for all cohort-specific curation and cross-cohort de-duplication rules from the publication.

The example notebook at [examples/subject_leakage_demo.ipynb](examples/subject_leakage_demo.ipynb) uses simulated participants only. It demonstrates the method; it does not regenerate manuscript results.

## Reproducibility and limitations

- A synthetic input and all Python unit tests can run without downloading biomedical data.
- Real-cohort execution requires separately obtained, appropriately licensed curation and abundance matrices.
- Reported full-study results include additional cohort-specific cleaning, nested tuning, simulations, cross-cohort comparisons, and R-based benchmarks. These are **not yet reproduced** by this minimal public repository.
- Grouped CV addresses sharing an individual between train and test. It does not resolve differences between independent cohorts, batch effects, or changes in case definitions.
- Some small grouped folds cannot yield reliable AUC estimates; interpretation requires subject-level uncertainty.

Source code does not contain the manuscript, submissions, internal review records, account credentials, or patient-level records.

## Testing

```bash
python -m pytest -q
```

GitHub Actions runs the Python test suite on pushes and pull requests. Those tests verify module behavior and synthetic pipelines, not the complete published analysis.

Code version: 0.3.1.
