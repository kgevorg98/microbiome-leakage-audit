"""Load locally exported curatedMetagenomicData tables and R abundance matrices.

No participant-level data are distributed in this repository.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_cohort(data_dir, cohort, label="disease"):
    """Return X, y, subject IDs and feature names for one cohort.

    Data directory must contain curation/<cohort>.tsv and
    matrices/<cohort>.relative_abundance.rda as exported by the study.
    """
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("Install analysis requirements (pyreadr is required)") from exc
    root = Path(data_dir)
    metadata = pd.read_csv(root / "curation" / f"{cohort}.tsv", sep="\t", low_memory=False).set_index("sample_id")
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
    if cohort == "YassourM_2018" and label == "sex":
        newborn = metadata.age_group.astype(str).eq("Newborn")
        abundance, metadata, target = abundance.loc[newborn], metadata.loc[newborn], target.loc[newborn]
    subjects = metadata.subject_id.astype(str)
    hashes = abundance.apply(lambda row: hashlib.md5(row.to_numpy().tobytes()).hexdigest(), axis=1)
    duplicate = hashes.duplicated(keep="first") & hashes.map(subjects.groupby(hashes).nunique()).eq(1)
    abundance, target, subjects = abundance.loc[~duplicate], target.loc[~duplicate], subjects.loc[~duplicate]
    return abundance.to_numpy(), target.astype(int).to_numpy(), subjects.to_numpy(), abundance.columns.to_list()
