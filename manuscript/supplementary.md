---
title: "Supplementary Material"
subtitle: "{{TITLE}}"
---

# Supplementary Methods

## S1. Data, labels and patient grouping

FedBCa (Zenodo record 10409145, file `FedBCa.rar`, MD5 a0910eaeb00a7be33ee2cf15e90f4506) contains, for each of four centres, axial T2WI volumes (NIfTI), one manual tumour mask per tumour and a spreadsheet that links each mask to its image and to a binary MIBC label. We verified every image–mask pair for matching size, spacing, origin and direction (no mismatch), recorded acquisition geometry, and counted mask connected components. Tumours that share an image file belong to one patient; this yields {{N_PATIENTS}} patients for {{N_TUMOURS}} tumours ({{N_MULTI}} patients with more than one tumour; up to {{MAX_TPP}} tumours per patient). The patient counts per centre (119, 47, 24 and 31) equal those implied by Table 1 of the dataset publication; the text of that publication reports 228 patients, which we could not reproduce from the released files. Different image files never contained identical or near-identical volumes (maximum correlation of down-sampled volumes {{MAX_DUP_R}}). The released files contain no clinical variables, reader assessments (e.g. VI-RADS) or follow-up; per-centre age, sex, T stage and grade reported in the dataset publication are shown in Table 1 of the main text. The T-stage counts of the publication (pT2–T4: 30, 25, 13 and 17 tumours in C1–C4) match the MIBC labels of the released files.

## S2. Pre-processing

All steps are label-free and applied to every scan identically, so they cannot leak information between training and test data.

- **Bias-field correction.** N4 [@n4] on a 4 × 4 × 1 shrunk image, Otsu foreground mask, 50/40/30 iterations at three fitting levels; the estimated field was applied at full resolution.
- **Resampling.** Images were resampled in-plane to 0.7 × 0.7 mm with B-spline interpolation (negative values clipped to zero); masks with nearest-neighbour interpolation onto the same grid. Through-plane spacing was not changed (no interpolation of new slices).
- **Peritumoural ring.** On every tumour-bearing slice, the mask was dilated in-plane with a disk of four pixels (2.8 mm, the whole-pixel width closest to 3 mm) and the tumour itself and any other tumour of the same patient were removed.
- **Perturbed masks.** One-pixel in-plane erosion and dilation of the tumour mask (erosion was skipped when fewer than ten voxels would remain), with the ring recomputed from each perturbed mask.
- **2.5D crops (deep learning).** For every tumour-bearing slice, a square field of view centred on the tumour's in-plane bounding box (side = max(80 mm, box + 2 × 10 mm)) was cut from slices z − 1, z and z + 1 (three channels) and resized to 224 × 224 pixels. Intensities were scaled per scan to the 0.5th–99.5th percentile of the scan foreground.

## S3. Radiomics

PyRadiomics 3.1.0 [@pyradiomics] was run with `normalize: true` (scan-wise z-score, scale 100), `removeOutliers: 3`, `binWidth: 5`, `force2D: true` (axial plane), `minimumROISize: 10` and image types Original, Laplacian of Gaussian (σ = 1, 2 and 3 mm) and Wavelet (four 2D sub-bands). Feature classes were 3D shape (intratumoural region only), first-order, GLCM, GLRLM, GLSZM, GLDM and NGTDM, giving {{N_FEAT_IT}} intratumoural and {{N_FEAT_PT}} peritumoural features after exclusion of the {{N_PLACEHOLDER_FEAT}} NGTDM coarseness features in which PyRadiomics returned its fixed placeholder for an undefined value (10⁶; {{N_PLACEHOLDER_TUM}} tumours). This label-free exclusion was added after the first run of the leakage audit (S6). The strict models of the first run had never selected these features; excluding them changed the repeat-averaged S1 AUC of the primary radiomics model from {{RUN1_A_S1_RAD}} to {{NOW_A_S1_RAD}} (largest change for any S1 model {{RUN1_MAXDIFF_S1}}; S2 and S3 at most {{RUN1_MAXDIFF_S23}}). The parameter files are part of the code release. Feature robustness was measured as ICC(3,1) across the original, eroded and dilated masks; the ICC was computed on the training tumours of each split only.

**Selection pipeline (inside each training split).** (1) remove features with standard deviation ≤ 10⁻⁸ or with one value in ≥ 95% of tumours; (2) keep features with ICC ≥ 0.75; (3) z-score; (4) keep features with univariable Mann–Whitney *p* < 0.05 (at least 20, ranked by |AUC − 0.5|); (5) greedy removal of features with |Pearson r| > 0.9 to an already kept, stronger feature; (6) L1-penalised logistic regression (liblinear, balanced class weights) with the inverse penalty C chosen from 12 log-spaced values (0.01–10) by grouped inner three-fold cross-validation using the one-standard-error rule; at most 10 features (largest |coefficient|) were retained.

## S4. Classifiers and hyper-parameters

All classifiers used balanced class weights (or the equivalent positive-class weight) and were tuned by grid search with grouped inner three-fold cross-validation on the training data, maximising the AUC (Supplementary Table S4). The pre-specified primary classifier for radiomics and fusion was L2-penalised logistic regression (C ∈ {0.01, 0.1, 1, 10}); support vector machine, k-nearest neighbours, random forest, extra trees, XGBoost, LightGBM and multilayer perceptron were secondary. The classification threshold of every model was the Youden-optimal point of grouped inner cross-validated predictions on the training data and was then applied unchanged to the test data.

## S5. Frozen backbones and the fine-tuned CNN

**Frozen backbones.** ResNet-50 (timm `resnet50.tv2_in1k`), DINOv2 ViT-B/14 (timm `vit_base_patch14_dinov2.lvd142m`, 224-pixel input) and the BiomedCLIP ViT-B/16 vision encoder (Hugging Face `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) were used without any training. Weight files were verified by SHA-256 (listed in the code release). Each crop was normalised with the backbone's own mean and standard deviation; the tumour embedding was the mean over its five largest-area slices. The linear probe (standardisation → PCA with 5, 10, 20 or 40 components → L2-logistic regression, C ∈ {0.001, 0.01, 0.1, 1}) was tuned by grouped inner cross-validation.

**Fine-tuned CNN.** ResNet-18 (timm `resnet18.tv_in1k`) with a single-logit head was trained on crops resized to 128 × 128 pixels (≈ 0.63 mm per pixel at the 80-mm field of view), using up to seven slices per tumour with ≥ 30% of that tumour's maximum slice area. Optimisation: AdamW (learning rate 10⁻⁴ for the backbone and 10⁻³ for the head, weight decay 10⁻⁴), one-cycle schedule, batch size 32, binary cross-entropy weighted by the inverse class ratio, at most 30 epochs. Augmentation (applied on the fly): horizontal flip, rotation ±15°, scaling ±10%, translation ±5%, gamma 0.8–1.2 and gain 0.9–1.1. Twenty per cent of the training patients (stratified by centre and label) formed the validation set; training stopped after seven epochs without improvement of the validation tumour-level AUC and the best-validation weights were restored. The tumour score was the mean slice logit. Training used PyTorch on an Apple-silicon GPU (MPS). The test fold was scored after every epoch only to quantify the effect of choosing the epoch on the test data (audit L5); these scores never influenced training.

## S6. Leakage audit: implementation

Each variant changed exactly one step of the strict pipeline and reused its code, seeds and splits.

- **L1 Slice-level split (CNN).** All 2.5D slices were split at random into ten folds irrespective of tumour or patient, and performance was evaluated per slice, as in studies that report slice-level accuracy. The early-stopping validation set was also drawn at slice level.
- **L2 Patient grouping ignored.** Outer and inner folds were stratified by label at the tumour level, so tumours of one patient could fall in both training and test folds.
- **L3 Feature selection on all data.** The full selection pipeline (S3) was run once on all tumours; the selected features were then used in the otherwise strict cross-validation.
- **L4 Oversampling before splitting.** SMOTE [@smote] (k = 5) balanced the classes on the full radiomics matrix; synthetic tumours were then distributed across folds and evaluated together with the real ones. The reference was the identical strict pipeline without the ICC filter, because synthetic tumours have no perturbed masks.
- **L5 Best of many on the test set.** Radiomics: every combination of three feature budgets (5, 10, 20), eight classifiers and their full hyper-parameter grids ({{N_L5_CONFIGS}} configurations) was fitted on each training fold of the first two S1 repeats, and the configuration with the highest mean test AUC was reported. CNN: the epoch with the highest test AUC (among the epochs trained before early stopping) was reported instead of the epoch chosen on the validation set, both in patient-grouped cross-validation and in leave-one-centre-out validation, where the test AUC is that of the held-out centre. To show how well the validation set guided epoch choice, we computed in each fold the Spearman correlation between validation and test AUC across epochs and report the median over folds.
- **L6 Threshold on the test set.** The Youden threshold was re-derived from the test predictions themselves (affects threshold metrics only).
- **L7 Transductive adaptation.** ComBat [@johnson; @fortin] (neuroCombat 0.2.12) harmonised radiomics features by centre either on the training centres only (strict) or on all centres including the test centre, without (label-free transduction) or with the MIBC label as a protected covariate (label leakage). For the CNN, batch-normalisation statistics were re-estimated on the unlabelled test-centre images (AdaBN [@adabn]). **Why ComBat with the outcome as covariate can invert predictions.** With the label as a protected covariate, ComBat estimates the label effect from all tumours (including the test centre), rescales each centre's residuals and then adds the label effect back. Wherever a centre's scale estimate departs from the pooled one, the harmonised feature therefore carries a term that is a deterministic function of the outcome. In pooled cross-validation its net effect on the AUC was small (Table 3). Under leave-one-centre-out validation, the model for held-out centre {{L7H3_MIN_CENTRE}} selected {{L7H3_NSEL}} feature ({{L7H3_FEAT}}), in which one {{L7H3_OUTLIER_LABEL}} tumour of that centre had a value about {{L7H3_RATIO}} times the training median. This value dominated both the pooled label effect and the scale estimates. Before harmonisation, the feature's within-centre AUCs were {{L7H3_FEAT_AUC_RAW}}; afterwards it separated MIBC from NMIBC with an AUC of {{L7H3_FEAT_AUC_TRAIN}} in each training centre but with the opposite sign in {{L7H3_MIN_CENTRE}} (AUC {{L7H3_FEAT_AUC_TEST}}). All predictions in {{L7H3_MIN_CENTRE}} were therefore inverted (model AUC {{L7H3_MIN_AUC}}; other held-out centres {{L7H3_OTHER_RANGE}}). The first run of the audit showed the same inversion in {{RUN1_H3_CENTRE}} through a PyRadiomics placeholder value (S3), so the artefact arises from fitting ComBat with the outcome in the presence of an extreme value, not from one particular feature.
- **L8 Centre confounding.** (i) the CENTRE model, which assigns each tumour the MIBC prevalence of its centre in the training folds; (ii) the prevalence-weighted mean of within-centre AUCs compared with the pooled S1 AUC; (iii) centre identifiability from image features (S7).

## S7. Domain-shift analysis

Centre identifiability was quantified with a four-class multinomial logistic regression (C = 0.1, standardised inputs; 50 principal components for deep features) in patient-grouped, centre-stratified five-fold cross-validation, reporting the macro-averaged one-versus-rest AUC and accuracy (chance accuracy 0.25 for balanced classes; {{CENTRE_MAJORITY}} for always predicting the largest centre). Two-dimensional t-SNE maps (perplexity 30, on 30 principal components) were computed for visualisation only.

## S8. Statistical analysis

**Confidence intervals.** Patients (not tumours) were resampled with replacement 2,000 times; all tumours of a resampled patient entered the bootstrap sample. Percentile intervals are reported. For S1, the AUC of each bootstrap sample was averaged over the five repeats (pooled out-of-fold predictions per repeat), so the interval reflects patient sampling, not split variation; fold-level means and standard deviations are given in Supplementary Table S5. Paired differences between models used the same resamples. **Random-effects pooling.** Per-centre LOCO AUCs were logit-transformed with standard errors from the cluster bootstrap and pooled with DerSimonian–Laird τ² [@dersimonian]; the 95% CI used the Hartung–Knapp–Sidik–Jonkman adjustment (t with k − 1 degrees of freedom, never narrower than the DerSimonian–Laird interval) [@inthout] and the 95% prediction interval used t with k − 2 degrees of freedom [@riley2016; @debray]. With four centres these intervals are wide by construction. **Calibration** was summarised by calibration-in-the-large (logistic intercept with the logit of the prediction as offset), calibration slope and Brier score [@vancalster]. **Decision curves** [@vickers2006; @vickers2008] plot net benefit = TP/n − FP/n × pt/(1 − pt) against threshold probability pt. To separate calibration from discrimination, the pooled S2 predictions of the radiomics and size models were also recalibrated (logistic regression of the outcome on the logit of the prediction) and their decision curves recomputed; because this step uses the test labels, it explains differences in net benefit but is not a performance estimate. Across thresholds of 0.05–0.60, the difference in net benefit (radiomics minus size) ranged from {{DCA_RAD_SIZE_GAPRANGE_RAW}} before and from {{DCA_RAD_SIZE_GAPRANGE_RECAL}} after recalibration. **Sample size.** With {{N_MIBC}} MIBC events among {{N_TUMOURS}} tumours (prevalence {{PREV}}), Riley's criterion for a global shrinkage factor ≥ 0.9 [@riley2020] implies that a model with ten candidate predictors would need about {{RILEY_N10}} tumours if the anticipated Cox–Snell R² is {{RILEY_R2}} (the value corresponding to a C statistic of 0.80 at this prevalence). Our data therefore support at most a handful of predictors; we capped selection at ten, used penalised estimation, and treat all results as exploratory. For external validation, {{N_MIBC}} events are below the ≥ 100 events usually recommended for precise calibration estimates [@riley2021].

## S9. Use of a large language model

A large language model (Claude, Anthropic) was used only for language editing of the manuscript. It was not used to generate code, run analyses, or draft scientific content. All numerical results in the manuscript are inserted automatically from the analysis outputs (`manuscript/values_results.tsv`) rather than transcribed. The authors take full responsibility for the content.

<!-- PAGEBREAK -->

# Supplementary Tables

Tables S1–S4 are reproduced below. All supplementary tables (S1–S12) are provided in machine-readable form in the workbook *Supplementary_Tables.xlsx*, one sheet per table:

- **Table S5.** Performance of all models (primary and secondary) under S1, S2 and S3: AUC with 95% CI, fold-level mean and SD (S1), sensitivity, specificity, PPV, NPV, balanced accuracy, Brier score, calibration intercept and slope.
- **Table S6.** Per-centre AUCs under S2 (leave-one-centre-out) and S3 (development on C1), with MIBC prevalence and mean predicted risk.
- **Table S7.** Random-effects pooling of per-centre S2 AUCs (logit scale): DerSimonian–Laird and Hartung–Knapp–Sidik–Jonkman 95% CIs, 95% prediction intervals, τ², I² and Cochran's Q.
- **Table S8.** Paired cluster-bootstrap AUC differences versus the tumour-size comparator, with unclustered DeLong *p* values for reference (S2 and S3 only; in S1 each tumour has one prediction per repeat).
- **Table S9.** Leakage audit: every strict reference and leaky variant, with AUC, 95% CI and paired difference.
- **Table S10.** Audit L5: test AUC of every radiomics configuration (feature budget × classifier × hyper-parameters).
- **Table S11.** Centre identifiability from image features (macro AUC and accuracy of a four-class centre classifier).
- **Table S12.** Final radiomics model refitted on all tumours: selected features, coefficients, mean |SHAP|, selection frequency across the S1 training folds, and Spearman correlation with log tumour volume.

## Table S1. TRIPOD+AI checklist

Items are paraphrased; see the TRIPOD+AI statement for the full wording.

<!-- TABLE: manuscript/checklists/TRIPOD_AI.tsv -->

<!-- PAGEBREAK -->

## Table S2. CLEAR checklist

<!-- TABLE: manuscript/checklists/CLEAR.tsv -->

<!-- PAGEBREAK -->

## Table S3. Software versions and pretrained weights

<!-- TABLE: results/tables/TableS3_software.csv -->

## Table S4. Fixed settings and hyper-parameter search spaces

<!-- TABLE: results/tables/TableS4_hyperparameters.csv -->

<!-- PAGEBREAK -->

# Supplementary references

<!-- REFERENCES -->
