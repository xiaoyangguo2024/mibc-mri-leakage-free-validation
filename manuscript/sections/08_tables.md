# Tables

**Table 1** Characteristics of the four centres

<!-- TABLE: results/tables/Table1_cohort.csv -->

Rows marked † are reported per tumour in the dataset publication [@fedbca] and are not part of the released files; all other rows were computed from the released images, masks and labels. Patients were identified as tumours that share one scan. IQR interquartile range, MIBC muscle-invasive bladder cancer, NMIBC non-muscle-invasive bladder cancer, TE echo time, TR repetition time

<!-- PAGEBREAK -->

**Table 2** Performance of the pre-specified primary models under the three leakage-free protocols

<!-- TABLE: results/tables/Table2_strict.csv -->

AUCs with 95% CIs from 2,000 patient-level cluster bootstrap resamples. S1: out-of-fold predictions pooled within each of five repeats, AUC averaged over repeats. ΔAUC: paired difference from the tumour-size model on the same tumours. Pooled per-centre AUC: random-effects estimate over the four held-out centres on the logit scale (DerSimonian–Laird τ², Hartung–Knapp–Sidik–Jonkman CI) with the 95% prediction interval (PI) for a new centre. Sensitivity and specificity at the Youden threshold fixed on the training data. ‡ Evaluated on the first S1 repeat only. AUC area under the ROC curve, CI confidence interval, CNN convolutional neural network, IT intratumoural, PT peritumoural

<!-- PAGEBREAK -->

**Table 3** Leakage audit: the same models evaluated with one evaluation flaw at a time

<!-- TABLE: results/tables/Table3_audit.csv -->

Each flawed variant changes one step of the strict pipeline; its ΔAUC is the paired cluster-bootstrap difference from the reference listed above it (95% CI). Slice-level rows are scored per slice; all other rows per tumour. L4 is evaluated on real and synthetic tumours, so no paired difference is given; its reference is the strict pipeline without the ICC filter. Fold means: mean of the per-fold test AUCs, without CI. L6 (threshold chosen on the test set) does not change the AUC and is reported in the text. ComBat, empirical Bayes harmonisation by centre; AdaBN, batch-normalisation statistics re-estimated on the test-centre images; LOCO leave-one-centre-out
