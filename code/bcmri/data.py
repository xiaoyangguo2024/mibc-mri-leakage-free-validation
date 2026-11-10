"""Load the cohort and feature tables in a consistent order (index = case_id)."""
import numpy as np
import pandas as pd

from . import config as C
from .stats import icc_per_feature

BACKBONES = ["ResNet50_IN", "DINOv2_B", "BiomedCLIP"]
SIZE_COLS = ["IT_original_shape_VoxelVolume", "IT_original_shape_Maximum2DDiameterSlice"]


def load_cohort():
    coh = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    coh["y"] = coh["label"].astype(int)
    return coh


PLACEHOLDER = 1e6        # PyRadiomics returns NGTDM Coarseness = 10^6 when the NGTDM is empty (value undefined)


def _read_radiomics(version):
    it = pd.read_csv(C.PROCESSED / f"radiomics_IT_{version}.csv", index_col="case_id").add_prefix("IT_")
    pt = pd.read_csv(C.PROCESSED / f"radiomics_PT_{version}.csv", index_col="case_id").add_prefix("PT_")
    return it.join(pt)


def placeholder_columns():
    """Coarseness features that hold PyRadiomics' 10^6 placeholder (an undefined value) in any tumour
    of the original-mask extraction."""
    R = _read_radiomics("orig")
    return [c for c in R.columns if c.endswith("_ngtdm_Coarseness") and (R[c] == PLACEHOLDER).any()]


def load_radiomics(version="orig"):
    """IT_ and PT_ features without the features that contain undefined values (placeholder columns are
    dropped in every mask version, so all analyses share one candidate set). Label-free; added after the first
    audit run showed that ComBat amplified one placeholder into an artefact (ANALYSIS_PLAN_en.md, deviations)."""
    return _read_radiomics(version).drop(columns=placeholder_columns())


def load_deep(name):
    return pd.read_csv(C.PROCESSED / f"deep_{name}.csv", index_col="case_id")


class ICCProvider:
    """ICC(3,1) across original / eroded / dilated masks, computed on the cases it is given
    (pass the training cases of a fold so the filter never sees test images)."""

    def __init__(self):
        self.v = {k: load_radiomics(k) for k in ("orig", "ero", "dil")}
        cols = self.v["orig"].columns
        for k in ("ero", "dil"):
            self.v[k] = self.v[k].reindex(columns=cols)

    def __call__(self, ids, cols):
        stack = np.stack([self.v[k].loc[ids, cols].values for k in ("orig", "ero", "dil")], axis=1)
        return pd.Series(icc_per_feature(stack), index=cols)


def size_features(rad):
    X = rad[SIZE_COLS].copy()
    X["IT_original_shape_VoxelVolume"] = np.log(X["IT_original_shape_VoxelVolume"].clip(lower=1))
    X.columns = ["log_volume", "max_2d_diameter"]
    return X
