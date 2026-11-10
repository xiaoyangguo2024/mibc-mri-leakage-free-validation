# Materials and methods

## Study design and data

This retrospective study analysed only a public, de-identified dataset, so institutional review board approval and informed consent were not required; the ethics committees of the four contributing hospitals approved the original data collection and waived informed consent [@fedbca]. We used the FedBCa dataset [@fedbca] (Zenodo record 10409145; CC BY 4.0), which contains axial T2-weighted images (T2WI) acquired between November 2019 and July 2022 on four 3.0-T scanners: MAGNETOM Skyra and MAGNETOM Verio (Siemens), uMR 780 (United Imaging Healthcare) and Discovery MR750w (GE Healthcare).

The data providers included patients with untreated bladder cancer, or diagnostic transurethral resection of bladder tumour (TURBT) only, confirmed by TURBT or cystectomy within 2 weeks after MRI. They excluded tumours without a pathological T stage, non-urothelial histology and recurrent tumours [@fedbca]. The reference standard was the histopathological T stage of the TURBT specimen (which included detrusor muscle at the tumour base) or of the cystectomy specimen: MIBC (≥ pT2) versus NMIBC (Ta or T1). One radiologist segmented each tumour on T2WI and a second radiologist reviewed the masks; both had 14 years of experience [@fedbca].

We included every released tumour. The tumour was the unit of analysis and the patient the grouping unit: tumours sharing one scan belong to one patient, and every split kept all tumours of a patient on the same side. No clinical variables are released. The analysis plan was written before any model was trained and is released with the code; it pre-specified the primary models, protocols and endpoints. Reporting follows TRIPOD+AI [@tripod_ai] and CLEAR [@clear] (Supplementary Tables S1 and S2).

## Image pre-processing

Images underwent N4 bias-field correction [@n4] and in-plane resampling to 0.7 × 0.7 mm; slice spacing was kept as acquired. A 3-mm peritumoural ring was obtained by in-plane dilation and excluded the tumour and any other tumour of the same patient. Masks eroded and dilated by one voxel in-plane served to measure feature robustness to segmentation (Supplementary Methods S2).

## Models

Four model families and two comparators were pre-specified (Fig. 1a). In each family the primary model was named before training; all other variants were secondary.

**Radiomics.** PyRadiomics [@pyradiomics], whose feature definitions largely follow the Image Biomarker Standardisation Initiative [@ibsi], extracted two-dimensional features from the tumour (intratumoural, IT; {{N_FEAT_IT}} features) and from the ring (peritumoural, PT; {{N_FEAT_PT}} features) on original, Laplacian-of-Gaussian and wavelet images; features holding PyRadiomics' placeholder for undefined values were excluded (Supplementary Methods S3). Within each training split, features were filtered for near-zero variance, segmentation robustness (ICC(3,1) ≥ 0.75 across original, eroded and dilated masks), univariable association (Mann–Whitney *p* < 0.05) and redundancy (|r| > 0.9). L1-penalised logistic regression, with its penalty chosen by grouped inner cross-validation and the one-standard-error rule, then retained at most 10 features. The primary radiomics model (RAD) combined IT and PT features with L2-penalised logistic regression. Secondary variants used IT features only or one of seven other classifiers, each tuned by grouped inner three-fold cross-validation (Supplementary Methods S3 and S4).

**Deep learning.** The primary deep-learning model was an ImageNet-initialised 2.5D ResNet-18 [@resnet] fine-tuned on tumour-centred crops (three adjacent slices as channels, 128 × 128 pixels, field of view ≥ 80 mm, up to seven slices per tumour) with a class-weighted loss and on-the-fly augmentation. Training stopped on the AUC of a validation set formed by 20% of the training patients; the tumour score was the mean slice logit. Secondary deep models were linear probes on embeddings from three frozen backbones (ImageNet ResNet-50 [@resnet], DINOv2 ViT-B/14 [@dinov2] and BiomedCLIP ViT-B/16 [@biomedclip]), averaged over each tumour's five largest slices (Supplementary Methods S5).

**Fusion and comparators.** Deep-radiomics fusion (DLR, primary fusion model) added up to 20 principal components of the best frozen backbone to the radiomics pool before selection; late fusion averaged the RAD and probe probabilities. The SIZE comparator was a logistic regression on log tumour volume and maximum axial diameter. The CENTRE model assigned each tumour the MIBC prevalence of its centre in the training data; it uses no image information and served only to expose centre confounding.

## Leakage-free evaluation protocols

- **S1 (internal):** patient-grouped, centre-by-label stratified five-fold cross-validation, repeated five times.
- **S2 (external, primary endpoint):** leave-one-centre-out (LOCO); each centre in turn was held out and models were developed on the other three.
- **S3 (single-centre development):** models developed on the largest centre (C1) and tested on C2–C4.

Every data-dependent step was fitted on training data only: feature filtering and selection, scaling, principal components, hyper-parameters, early stopping and the classification threshold (Youden point of grouped inner cross-validated predictions). Because of computing cost, the CNN was evaluated on the first S1 repeat, and comparisons with the CNN used that repeat.

## Leakage audit

The same models were re-evaluated with one common flaw introduced at a time, everything else unchanged (Table 3; Supplementary Methods S6):

- L1: slice-level random split;
- L2: tumour-level split that ignores patients;
- L3: feature selection on the whole dataset before cross-validation;
- L4: SMOTE [@smote] oversampling before splitting;
- L5: reporting the best of many configurations, or of many training epochs, by test AUC;
- L6: threshold chosen on the test set;
- L7: transductive harmonisation, i.e. ComBat [@johnson; @fortin] fitted on all centres (with or without the label as covariate) or batch-normalisation statistics re-estimated on the test centre (AdaBN [@adabn]);
- L8: centre confounding in pooled cross-validation, quantified by the CENTRE model, within-centre AUC and centre identifiability.

Inflation was the leaky AUC minus the strict AUC on the same tumours.

## Domain shift and interpretability

Centre identifiability was the macro one-versus-rest AUC of a four-class logistic regression predicting the centre from radiomics or deep features (patient-grouped five-fold cross-validation); t-SNE maps were drawn for visualisation. For the RAD model refitted on all tumours we computed exact linear SHAP values, and we correlated each selected feature with log tumour volume (Spearman). For the CNN, we compared the share of Grad-CAM [@gradcam] energy inside the tumour and the ring with each region's share of the image, and measured the AUC after replacing the tumour, the ring or everything outside the tumour by a blurred copy.

## Statistical analysis

Discrimination was measured by the tumour-level area under the ROC curve (AUC), with 95% confidence intervals (CIs) from 2,000 patient-level cluster bootstrap resamples. For S1, out-of-fold predictions were pooled within each repeat and the AUC averaged over repeats. Models were compared by paired cluster-bootstrap AUC differences; DeLong tests [@delong; @sun_xu], which ignore clustering, are given for reference only. Per-centre LOCO AUCs were pooled on the logit scale with a random-effects model [@dersimonian; @debray], reporting the Hartung–Knapp–Sidik–Jonkman CI [@inthout], the 95% prediction interval and I². Calibration was summarised by calibration-in-the-large, calibration slope and Brier score [@vancalster], and clinical utility by decision curve analysis [@vickers2006] for threshold probabilities of 0.05–0.60. With {{N_MIBC}} MIBC events, selection was capped at 10 predictors [@riley2020]. Analyses used Python 3.11 with scikit-learn [@sklearn], PyTorch and timm [@timm] (versions in Supplementary Table S3; random seed 20261109).

## Use of large language models

A large language model (Claude, Anthropic) was used only for language editing of the manuscript. It was not used to generate code, run analyses, or draft scientific content. The authors take full responsibility for the content.
