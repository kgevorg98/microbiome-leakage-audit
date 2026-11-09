"""Run the standalone leakage audit on a locally obtained study cohort."""
import argparse
import json
from pathlib import Path

from leakcheck import audit
from analysis.curated_data import load_cohort
from analysis.preprocessing import classifier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--label", choices=["disease", "sex"], default="disease")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()
    X, y, subjects, features = load_cohort(args.data_dir, args.cohort, args.label)
    result = audit(lambda: classifier(args.C), X, y, subjects, n_folds=args.folds, seeds=range(args.seeds))
    result.update(cohort=args.cohort, label=args.label, n_species=len(features), C=args.C)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
