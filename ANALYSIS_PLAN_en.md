# Analysis plan — English summary

The binding plan is `分析方案.md` (Chinese). Version 1.0 was written after the data were downloaded and before any
model was trained; version 1.1 (2026-11-09 07:20) was revised after the data audit and **before any model was trained**.
This English summary was prepared afterwards, for readers; where it differs from the Chinese text, the Chinese text
applies. Deviations between the plan and the analyses actually run are listed at the end.

## Questions

1. **Strict performance.** Using T2WI only, how well do radiomics, frozen deep features, a fine-tuned CNN and fusion
   models separate MIBC from NMIBC under patient-grouped cross-validation and leave-one-centre-out (LOCO) validation,
   and do they beat a simple tumour-size baseline?
2. **Leakage.** How much do common evaluation flaws inflate the AUC of the same models on the same data, and can a
   standard ResNet reproduce the published AUCs of about 0.98–0.99?
3. **Centre shortcut.** Can image features identify the acquiring centre, and how much of pooled cross-validated AUC
   reflects centre rather than invasion?
4. **Interpretation and utility.** Where do the models look (tumour, peritumoural ring), and do decision curves show
   net benefit?

## Data

FedBCa (Zenodo 10409145, CC BY 4.0). Unit of analysis: tumour. Grouping unit: patient (tumours sharing a scan).
No clinical variables, reader assessments or follow-up are available; no clinical or survival models are fitted.
Data audit (v1.1): 275 tumours, 85 MIBC, 221 patients, 27 patients with more than one tumour.

## Pre-specified primary models (v1.1)

- Primary radiomics model: intratumoural + peritumoural features (RAD_ITPT) with L2-penalised logistic regression.
- Primary deep-learning model: fine-tuned 2.5D ResNet-18.
- Primary fusion model: deep-radiomics fusion (DLR) with L2-penalised logistic regression.
- Primary comparator: tumour-size model.

The other seven classifiers, the three frozen backbones and late fusion are secondary: reported, not used for
conclusions. A centre-prevalence model is used only to expose confounding.

## Protocols and endpoints

- S1: patient-grouped, stratified 5-fold CV repeated 5 times, inner 3-fold grouped tuning (internal validation).
- S2: leave-one-centre-out (**primary endpoint**): tumour-level AUC of pooled LOCO predictions and random-effects
  pooling of the four per-centre AUCs (logit scale, DerSimonian–Laird) with I² and 95% prediction interval.
- S3: development on the largest centre (C1), external testing on C2–C4.
- Secondary: sensitivity, specificity, accuracy, PPV, NPV (threshold fixed on training data, Youden), Brier score,
  calibration intercept and slope, calibration curves, decision curves (threshold 5–60%).
- Uncertainty: 2,000 patient-level cluster bootstrap resamples; paired cluster bootstrap for ΔAUC; DeLong for reference.
- Confirmatory comparisons limited to primary model vs tumour size and strict vs each leaky protocol; all other
  comparisons exploratory.

## Leakage audit

One flaw at a time, everything else unchanged: L1 slice-level random split; L2 patient grouping ignored; L3 feature
selection on all data; L4 oversampling before splitting; L5 choosing the best of many configurations (or epochs) by
test AUC; L6 threshold chosen on the test set; L7 transductive use of test-centre data (ComBat, domain adaptation);
L8 centre confounding in pooled CV. Combined scenario: L1 + L5. If a flaw does not inflate performance, this is
reported as such and the framing of the paper follows the results.

## Deviations from the plan

- Segmentation robustness used one-voxel erosion and dilation only; the random contour translation listed in the
  plan was not implemented.
- Deep-learning crops were intensity-scaled to the 0.5th–99.5th percentile of the scan foreground (plan: 1st–99th).
- The tumour-size model used log tumour volume and maximum 2D diameter (plan also listed maximum slice area, which is
  nearly collinear with the diameter).
- No medical-imaging-pretrained ResNet (e.g. RadImageNet) was added: only unofficial weight mirrors of unknown
  provenance were available.
- Eleven NGTDM coarseness features that hold PyRadiomics' fixed placeholder for an undefined value (10^6; two
  tumours) were excluded from all radiomics analyses. This label-free rule was added after the first run of the
  leakage audit, in which ComBat fitted with the label as covariate amplified one placeholder into an artefactual
  feature that inverted the predictions in one held-out centre. The strict models of the first run never selected
  these features. After the exclusion the same inversion recurred through another feature with one extreme value,
  so it is reported as a property of label-informed ComBat (Supplementary Methods S6). First-run outputs are kept
  in results/predictions/run1_before_placeholder_fix/.
- The fine-tuned CNN was evaluated on the first S1 repeat only (computing time); comparisons with the CNN use that repeat.
- Reporting follows TRIPOD+AI and CLEAR; the CLAIM 2024 checklist and a METRICS self-assessment, listed in the plan,
  were not completed by the analysts.
