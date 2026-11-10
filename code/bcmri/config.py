"""Paths, constants and the global random seed. Every numbered script imports this."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]          # project root (BLCA_MRI_MIBC/)
CODE = ROOT / "code"
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "zenodo_10409145" / "FedBCa.rar"     # untrusted download
EXTRACTED = RAW / "FedBCa_extracted"                 # untrusted, extracted archive
INTERIM = ROOT / "data" / "interim"                  # preprocessed volumes, crops
PROCESSED = ROOT / "data" / "processed"              # cohort table, features, splits
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDS = RESULTS / "predictions"
MODELS = RESULTS / "models"
LOGS = ROOT / "logs"
WEIGHTS = ROOT / "env" / "weights"                   # pretrained network weights

ARCHIVE_MD5 = "a0910eaeb00a7be33ee2cf15e90f4506"
ARCHIVE_BYTES = 1028155736

SEED = 20261109
CENTRES = ["C1", "C2", "C3", "C4"]
SCANNER = {"C1": "Siemens MAGNETOM Skyra", "C2": "United Imaging uMR 780",
           "C3": "GE Discovery MR750w", "C4": "Siemens MAGNETOM Verio"}
VENDOR = {"C1": "Siemens", "C2": "United Imaging", "C3": "GE", "C4": "Siemens"}

# Preprocessing
INPLANE_SPACING = 0.7      # mm, in-plane resampling target; slice thickness is kept
RING_MM = 3.0              # peritumoural ring width (in-plane)
CROP_MARGIN_MM = 10.0      # context added around the tumour bounding box for 2.5D crops
CROP_SIZE = 224
TOPK_SLICES = 5            # slices per tumour for frozen-feature extraction

N_BOOT = 2000
OUTER_FOLDS = 5
OUTER_REPEATS = 5
INNER_FOLDS = 3
MAX_FEATURES = 10

for _d in (INTERIM, PROCESSED, TABLES, FIGURES, PREDS, MODELS, LOGS, WEIGHTS):
    _d.mkdir(parents=True, exist_ok=True)

# Keep BLAS/OpenMP single-threaded unless a script raises it: the Mac has 8 GB RAM.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
