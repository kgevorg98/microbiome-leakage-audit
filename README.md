# leakcheck

`leakcheck` measures the difference between sample-wise and subject-grouped cross-validation in datasets with repeated observations per participant. It also reports repeat structure and label constancy.

The repository contains only the standalone Python module and its unit tests. Research manuscripts, internal audit reports, and participant-level data are not included.

## Requirements

Python 3.10+, NumPy, pandas and scikit-learn.

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Usage

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from leakcheck import audit

result = audit(
    estimator_factory=lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000)
    ),
    X=features,
    y=labels,
    subjects=subject_ids,
    n_folds=5
)
print(result)
```

The estimator factory must construct a fresh, unfitted model whose data-dependent preprocessing is fitted within cross-validation folds. Each class and grouping structure must permit the requested stratified folds.

The difference between random and grouped AUC is a diagnostic for validation design, not a correction factor. The exploratory `correct_auc` function is intentionally disabled by default because retrospective correction did not generalise reliably. External validation is still necessary when the intended application concerns new cohorts or settings.

Code version: 0.3.1. The repository does not contain the complete study reproduction workflow. It provides the independently tested diagnostic module.
