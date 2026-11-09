# leakcheck

`leakcheck` evaluates subject-level leakage in classifiers trained on repeated microbiome observations. Its principal diagnostic is the difference between sample-wise and subject-grouped cross-validation, accompanied by descriptive measures of within-subject replication.

This repository contains the standalone diagnostic, tests, a synthetic-data notebook, and a workflow for analyzing locally available species-level abundance profiles. It does **not** contain manuscript files, participant-level data, or the complete computational pipeline used to generate all published tables and figures (see Reproducibility and limitations).

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
python -m pytest -q
```

For analyses of exported `.rda` matrices install the analysis extra instead: `python -m pip install -e ".[test,analysis]"`.
`requirements.txt` and `requirements-analysis.txt` list the same dependencies for environments that do not install the package; in that case run commands from the repository root so that `leakcheck` and `analysis` are importable.

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
python -m analysis.run_cohort --data-dir local-data --cohort HMP_2019_ibdmdb --label disease --folds 5 --seeds 10 --C 0.1
python -m analysis.run_cohort --data-dir local-data --cohort HMP_2019_ibdmdb --label disease --nested
```

The first command reports random-split and subject-grouped AUC for one fixed model (C = 0.1). `--nested` adds the whole-pipeline comparison used for the study's pipeline inflation: C is tuned over a grid by inner cross-validation, with random inner and outer folds (leaky pipeline) or subject-grouped inner and outer folds (grouped pipeline), over 5 seeds.

The `analysis.preprocessing.PrevalenceCLR` transformer learns the feature prevalence filter on the training fold and then applies the CLR transformation to training and test samples consistently. The loader applies the study's sample selection and cleaning: stool samples with a defined label, removal of exact duplicate profiles within a subject, removal of near-identical profiles recorded under different subjects, C. difficile infection versus healthy for `VincentC_2016` disease, and restriction of `YassourM_2018` sex to the children. It does not detect participants shared between cohorts.

The example notebook at [examples/subject_leakage_demo.ipynb](examples/subject_leakage_demo.ipynb) uses simulated participants only. It demonstrates the method; it does not regenerate manuscript results.

## Reproducibility and limitations

- A synthetic input and all Python unit tests can run without downloading biomedical data.
- Real-cohort execution requires separately obtained curatedMetagenomicData curation tables and abundance matrices.
- With the study's input files, `analysis.run_cohort` reproduces the per-task same-model AUCs and inflation (C = 0.1) and, with `--nested`, the whole-pipeline AUCs and pipeline inflation for all 16 study tasks (agreement to floating-point precision; Python 3.11, scikit-learn 1.9.1).
- Not included here: the sibling-dose experiments, intraclass-correlation and inflation-model fits, simulations, bootstrap and random-forest robustness checks, cross-cohort analyses, R-based benchmarks and figure generation.
- Grouped CV addresses sharing an individual between train and test. It does not resolve differences between independent cohorts, batch effects, or changes in case definitions.
- Some small grouped folds cannot yield reliable AUC estimates; interpretation requires subject-level uncertainty.

Source code does not contain the manuscript, submissions, internal review records, account credentials, or patient-level records.

## Testing

```bash
python -m pytest -q
```

GitHub Actions installs the package and runs the test suite on pushes and pull requests. The tests use synthetic data; they verify module behavior, not the published numbers.

Code version: 0.3.1.
