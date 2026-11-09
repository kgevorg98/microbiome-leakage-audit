"""Load locally exported curatedMetagenomicData tables and R abundance matrices.

No participant-level data are distributed in this repository.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_cohort(data_dir, cohort, label="disease", pseudocount=1e-5, return_log=False):
    """Return X, y, subject IDs and feature names for one cohort.

    Data directory must contain curation/<cohort>.tsv and
    matrices/<cohort>.relative_abundance.rda as exported by the study.
    Cleaning follows the study: exact within-subject duplicates are removed
    (first kept), near-identical profiles shared by different subjects are
    removed, and YassourM_2018 sex is restricted to the children.
    With return_log=True a dict of cleaning counts is returned as a fifth value.
    """
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("Install analysis requirements (pyreadr is required)") from exc
    root = Path(data_dir)
    full_metadata = pd.read_csv(root / "curation" / f"{cohort}.tsv", sep="\t", low_memory=False)
    metadata = full_metadata.set_index("sample_id")
    abundance = next(iter(pyreadr.read_r(str(root / "matrices" / f"{cohort}.relative_abundance.rda")).values()))
    abundance = abundance.loc[["|s__" in s and "|t__" not in s for s in abundance.index]]
    sample_ids = [s for s in abundance.columns if s in metadata.index]
    abundance = abundance[sample_ids].T.astype(float)
    metadata = metadata.loc[sample_ids]
    stool = metadata["body_site"].astype(str).str.lower().str.contains("stool|feces|faec", na=False)
    if label == "sex":
        target = metadata["sex"].astype(str).str.lower().map({"male": 1., "female": 0.})
    elif label == "disease":
        text = metadata["disease"].astype(str)
        if cohort == "VincentC_2016":
            target = pd.Series(np.where(text.str.contains("Clostridium difficile", case=False, na=False), 1.,
                                        np.where(text.str.lower().eq("healthy"), 0., np.nan)), index=text.index)
        else:
            target = pd.Series(np.where(text.str.lower().eq("healthy"), 0.,
                                        np.where(text.str.strip().str.lower().isin(["nan", "none", "", "unknown", "not applicable"]), np.nan, 1.)), index=text.index)
    else:
        raise ValueError("label must be 'disease' or 'sex'")
    keep = stool & target.notna()
    abundance, metadata, target = abundance.loc[keep], metadata.loc[keep], target.loc[keep]
    if not metadata.subject_id.notna().all():
        raise ValueError("Missing subject identifiers")
    if not abundance.index.is_unique or not np.allclose(abundance.sum(axis=1), 100, atol=1):
        raise ValueError("Invalid sample IDs or relative-abundance totals")
    subjects = metadata.subject_id.astype(str)
    n_before = len(abundance)
    # Exact duplicate profiles within one subject (technical replicates): keep the first.
    hashes = abundance.round(10).apply(lambda row: hashlib.md5(row.to_numpy().tobytes()).hexdigest(), axis=1)
    same_subject = hashes.map(subjects.groupby(hashes).nunique()).eq(1)
    drop = set(abundance.index[hashes.duplicated(keep="first") & same_subject])
    # Near-identical profiles assigned to different subjects (CLR correlation > 0.999): drop both.
    log = np.log(abundance.to_numpy() + pseudocount)
    log -= log.mean(axis=1, keepdims=True)
    log = (log - log.mean(axis=1, keepdims=True)) / log.std(axis=1, keepdims=True)
    similarity = log @ log.T / log.shape[1]
    np.fill_diagonal(similarity, 0)
    ids = subjects.to_numpy()
    cross = {abundance.index[k] for i, j in np.argwhere(np.triu(similarity > 0.999, 1))
             if ids[i] != ids[j] for k in (i, j)}
    drop |= cross
    keep = ~abundance.index.isin(list(drop))
    abundance, target, subjects = abundance.loc[keep], target.loc[keep], subjects.loc[keep]
    cleaning = {"n_before_clean": n_before, "dropped_exact_dup_same_subject": len(drop - cross),
                "dropped_cross_subject_identical": len(cross), "n_after_clean": len(abundance)}
    if cohort == "YassourM_2018" and label == "sex":
        # Children only (mothers excluded): subjects whose first curated record is a newborn sample.
        # All samples of those children are kept, including later infant samples.
        first_group = full_metadata.groupby("subject_id").age_group.first()
        child = subjects.map(first_group).eq("Newborn")
        abundance, target, subjects = abundance.loc[child], target.loc[child], subjects.loc[child]
        cleaning["restricted_to_children"] = True
    out = (abundance.to_numpy(), target.astype(int).to_numpy(), subjects.to_numpy(), abundance.columns.to_list())
    return (*out, cleaning) if return_log else out
