# Radiomics and deep learning for muscle-invasive bladder cancer on MRI: how much survives leakage-free validation?

Analysis code, analysis plan, derived tables, out-of-fold predictions and manuscript sources for a leakage-aware, multicentre benchmark of T2-weighted MRI radiomics and deep learning for predicting muscle-invasive bladder cancer (MIBC).

**Manuscript (under review):** Guo X, Wu M, Sun Z, Liao Z, Zeng K, Wen B. *Radiomics and deep learning for muscle-invasive bladder cancer on MRI: how much survives leakage-free validation?* Under review at *Insights into Imaging*.

**Corresponding author:** Bo Wen, Department of Urology, Shenzhen Hospital of Integrated Traditional Chinese and Western Medicine, No. 528 Xinsha Road, Shajing Street, Bao'an District, Shenzhen, Guangdong 518104, China. E-mail: [tjwb001@126.com](mailto:tjwb001@126.com).

## Data source

This study uses only the public **FedBCa** dataset:

- Cao K, et al. A multicenter bladder cancer MRI dataset and baseline evaluation of federated learning in clinical application. *Scientific Data* 2024;11:1147. [doi:10.1038/s41597-024-03971-0](https://doi.org/10.1038/s41597-024-03971-0)
- Dataset deposit: Zenodo record [10409145](https://doi.org/10.5281/zenodo.10409145), licence **CC BY 4.0**

Raw MRI volumes, annotations and other imaging files are **not** redistributed in this repository. Please download `FedBCa.rar` from the original Zenodo record (MD5: `a0910eaeb00a7be33ee2cf15e90f4506`) and extract it locally as described below. Derived radiomics and deep-feature tables, out-of-fold predictions and figures are included and are released under CC BY 4.0, with attribution to the FedBCa investigators.

Tumour-centred MRI crops used for Grad-CAM panels (`data/processed/gradcam_examples.npz`) are also omitted, because they contain image data. They are written again by `code/10_interpret.py` after the imaging data have been obtained and preprocessed. The published Figure 5 files under `results/figures/` are already included.

## Environment

Python **3.11** (the analysis was run on 3.11.18). Package versions reported in the manuscript (Supplementary Table S3) include:

| Package | Version | Role |
|---|---|---|
| numpy | 1.26.4 | numerics |
| scipy | 1.17.1 | statistics |
| pandas | 3.0.6 | tables |
| scikit-learn | 1.9.1 | models, cross-validation, metrics |
| SimpleITK | 2.5.7 | image I/O, N4, resampling |
| pyradiomics | 3.1.0 (built from the release source archive) | radiomics features |
| torch | 2.14.1 | deep learning |
| torchvision | 0.29.2 | image transforms |
| timm | 1.0.31 | ResNet and DINOv2 backbones |
| open_clip_torch | 4.2.1 | BiomedCLIP encoder |
| xgboost | 3.4.2 | classifier |
| lightgbm | 4.7.0 | classifier |
| imbalanced-learn | 0.14.2 | SMOTE (audit L4 only) |
| neuroCombat | 0.2.12 | ComBat harmonisation |
| statsmodels | 0.15.0 | calibration models |
| shap | 0.48.0 | SHAP values |
| scikit-image | 0.20.0 | mask morphology |
| matplotlib | 3.11.2 | figures |

Pinned environment files:

- `env/environment_explicit.txt` — conda packages with explicit URLs (the original run used conda-forge on osx-arm64)
- `env/requirements_pip.txt` — packages installed with pip. PyRadiomics was built from the [v3.1.0 source release](https://github.com/AIM-Harvard/pyradiomics/archive/refs/tags/v3.1.0.tar.gz) (installed package metadata reports 3.0.1a1)

Create an environment, for example:

```bash
conda create -n bcmri python=3.11
conda activate bcmri
# install the conda-forge stack from env/environment_explicit.txt where platform-compatible,
# then:
pip install -r env/requirements_pip.txt
```

On platforms other than osx-arm64, prefer installing the versions above from conda-forge or pip rather than the explicit osx-arm64 URLs in `environment_explicit.txt`.

Pretrained backbone weights are downloaded from Hugging Face into `env/weights/`. `env/weights/MANIFEST.tsv` lists the revision and SHA-256 of every weight file used. Fine-tuned CNN checkpoint files (`results/models/*.pt`, about 45 MB each) are not included; they are written again by `code/07_cnn.py`.

`run_step.sh` uses `env/py/bin/python` if you recreate that prefix, otherwise `${PYTHON:-python3}`.

## How to reproduce

1. Download `FedBCa.rar` from [Zenodo 10409145](https://doi.org/10.5281/zenodo.10409145) into `data/raw/zenodo_10409145/` and extract it to `data/raw/FedBCa_extracted/`. Confirm the archive MD5 is `a0910eaeb00a7be33ee2cf15e90f4506`.
2. Create the environment as above and place pretrained weights in `env/weights/` (see the manifest).
3. Run the numbered steps in order:

```bash
bash run_step.sh 01_inventory
bash run_step.sh 02_preprocess
bash run_step.sh 03_radiomics
bash run_step.sh 04_crops
bash run_step.sh 05_deep_features
bash run_step.sh 06_models_strict S1 S2 S3
bash run_step.sh 07_cnn strict loco s3 slice10 tumour5
bash run_step.sh 08_audit_ml
bash run_step.sh 09_domain
bash run_step.sh 10_interpret a b
bash run_step.sh 11_summarise
bash run_step.sh 12_figures fig1 fig2 fig3 fig4 fig5 fig6
bash run_step.sh 13_tables
bash run_step.sh 14_values
bash run_step.sh 15_graphical_abstract
```

Logs are written to `logs/`. Steps 06 and 08 use `BCMRI_JOBS` worker processes (default 4).

4. Optional: rebuild the manuscript with `python3 -I manuscript/build.py` and `python3 -I manuscript/build_supp.py` (pandoc; typst for PDF).

Steps 05, 07 and 10 (part b) use an Apple-silicon GPU (PyTorch MPS) when available, otherwise the CPU. All randomness is seeded (`SEED = 20261109` in `code/bcmri/config.py`). GPU training is not bit-for-bit deterministic, so CNN results can differ slightly between runs and machines.

`data.load_radiomics()` drops 11 NGTDM coarseness features in which PyRadiomics returned its placeholder for an undefined value (10^6). This label-free rule was added after the first audit run (see the deviations in `ANALYSIS_PLAN_en.md`; first-run outputs in `results/predictions/run1_before_placeholder_fix/`).

## Pipeline

| Step | Script | Content |
|---|---|---|
| 01 | `code/01_inventory.py` | Data audit: image–mask consistency, patient grouping, duplicate-scan check |
| 02 | `code/02_preprocess.py` | N4 bias-field correction, in-plane resampling to 0.7 mm, peritumoural ring, eroded/dilated masks |
| 03 | `code/03_radiomics.py` | PyRadiomics features (intratumoural and peritumoural; original, eroded and dilated masks) |
| 04 | `code/04_crops.py` | 2.5D tumour-centred crops for deep learning |
| 05 | `code/05_deep_features.py` | Frozen embeddings: ResNet-50, DINOv2 ViT-B/14, BiomedCLIP ViT-B/16 |
| 06 | `code/06_models_strict.py` | Strict protocols S1 (patient-grouped 5×5 CV), S2 (leave-one-centre-out), S3 (develop on C1) |
| 07 | `code/07_cnn.py` | Fine-tuned 2.5D ResNet-18 (strict, LOCO with/without AdaBN, S3, audit variants L1/L2) |
| 08 | `code/08_audit_ml.py` | Leakage audit for radiomics (L2–L5, L7) |
| 09 | `code/09_domain.py` | Centre identifiability, t-SNE maps |
| 10 | `code/10_interpret.py` | Final radiomics model + SHAP; Grad-CAM, region occlusion |
| 11 | `code/11_summarise.py` | Cluster-bootstrap CIs, random-effects pooling, calibration, decision curves, audit summary |
| 12–14 | `code/12_figures.py`, `13_tables.py`, `14_values.py` | Figures, tables, and every number quoted in the text |
| 15 | `code/15_graphical_abstract.py` | Graphical abstract (numbers read from the result tables) |

Shared code is in `code/bcmri/` (`config.py` paths and seed, `data.py`, `ml.py`, `stats.py`, `figstyle.py`). Unit tests: `code/tests/test_core.py`. Radiomics settings: `code/params_radiomics_IT.yaml`, `code/params_radiomics_PT.yaml`.

The analysis plan (`分析方案.md`, in Chinese; v1.1 written before any model was trained) pre-specified the primary models, protocols and endpoints. `ANALYSIS_PLAN_en.md` summarises those elements in English.

## Directory structure

```
.
├── README.md
├── LICENSE                          # MIT (code)
├── ANALYSIS_PLAN_en.md
├── 分析方案.md
├── run_step.sh
├── make_code_archive.sh
├── code/                            # analysis scripts and bcmri library
├── data/
│   ├── external/                    # values transcribed from the FedBCa paper
│   └── processed/                   # cohort table, radiomics and deep features (no images)
├── env/
│   ├── environment_explicit.txt
│   ├── requirements_pip.txt
│   └── weights/MANIFEST.tsv         # pretrained-weight revisions and SHA-256
├── results/
│   ├── tables/                      # manuscript and supplementary tables
│   ├── predictions/                 # out-of-fold and external predictions
│   ├── figures/                     # SVG, PDF, TIFF, PNG
│   └── models/final_rad_itpt_lr.pkl # final radiomics model (small)
├── logs/
└── manuscript/                      # text sources, references, checklists, build scripts
```

## Licence

- **Code:** MIT (see `LICENSE`).
- **Derived tables, predictions and figures:** CC BY 4.0, following the source dataset. Please cite the FedBCa paper and this work if you reuse them.

## Contact

Questions about the code or the manuscript: Bo Wen, [tjwb001@126.com](mailto:tjwb001@126.com).
