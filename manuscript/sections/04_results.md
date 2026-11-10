# Results

## Cohort

All {{N_TUMOURS}} released tumours from {{N_PATIENTS}} patients were analysed, and {{N_MIBC}} ({{PREV_PCT}}) were MIBC (Fig. 1, Table 1). More than one tumour was present in {{N_MULTI}} patients (up to {{MAX_TPP}}). MIBC prevalence varied across centres ({{PREV_RANGE}}), and MIBC tumours were larger than NMIBC tumours (median volume {{VOL_MIBC}} vs {{VOL_NMIBC}} ml). No image, mask or label was missing.

## Leakage-free performance

In patient-grouped cross-validation (S1), the primary radiomics, CNN and fusion models reached AUCs of {{AUCS_S1_RAD}}, {{AUCS_S1_CNN}} and {{AUCS_S1_DLR}}; tumour size alone reached {{AUCS_S1_SIZE}} (Table 2). On held-out centres (S2, primary endpoint), the AUCs were {{AUCS_S2_RAD}} for radiomics, {{AUCS_S2_DLR}} for fusion, {{AUCS_S2_CNN}} for the CNN and {{AUCS_S2_SIZE}} for size (Fig. 2a). Compared with size, ΔAUC was {{DSIZE_S2_RAD}} for radiomics, {{DSIZE_S2_DLR}} for fusion and {{DSIZE_S2_CNN}} for the CNN.

Per-centre AUCs of the radiomics model were {{CENRANGE_S2_RAD}}; the random-effects pooled AUC was {{RE_RAD}}, with a 95% prediction interval of {{PI_RAD}} for a new centre (Fig. 2b). When models were developed in the largest centre alone (S3), radiomics reached {{A_S3_RAD}}, the CNN {{A_S3_CNN}} and size {{A_S3_SIZE}}. Across all models and classifiers, the highest S2 AUC was {{S2_MAX_ANY}}, for {{S2_MAX_NAME}}, and no model was significantly better than size ({{S2_MAX_NAME}} ΔAUC {{DSIZE_S2_MAX}}; Fig. 2c; Supplementary Tables S5 and S8).

## Calibration and clinical utility

On held-out centres, calibration slopes were {{SLOPE_S2_RAD}} for radiomics, {{SLOPE_S2_CNN}} for the CNN and {{SLOPE_S2_SIZE}} for size (Fig. 6a). The mean predicted risk of the radiomics and size models followed the ranking of centre prevalence but overestimated it in the low-prevalence centre C1 (radiomics {{MEANP_S2_RAD_C1}}, size {{MEANP_S2_SIZE_C1}}; prevalence {{PREV_C1}}), whereas the CNN predicted its highest mean risk in C1 ({{MEANP_S2_CNN_C1}}; Fig. 6b). Radiomics gave a higher net benefit than treating all tumours as MIBC at thresholds of {{DCA_RAD_RANGE}}, and than size at {{DCA_RAD_GT_SIZE_RANGE}} (by up to {{DCA_RAD_SIZE_MAXGAP}}; Fig. 6c). This gap reflected calibration rather than discrimination: after logistic recalibration, radiomics exceeded size by at most {{DCA_RAD_SIZE_MAXGAP_RECAL}} (Supplementary Methods S8).

## Leakage audit

Splitting slices at random raised the CNN slice-level AUC from {{CNN_SLICE_A}} to {{CNN_L1_A}} ({{CNN_L1_D}}; Table 3, Fig. 3a, b). Splits that ignored patient grouping changed the AUC little: {{RAD_L2_DS}} for radiomics and {{CNN_L2_DS}} for the CNN. For radiomics, selecting features on all tumours added {{RAD_L3_D}}, and reporting the best of {{N_L5_CONFIGS}} configurations added {{RAD_L5_D}} (median configuration {{L5_MEDIAN}}; Fig. 3c). SMOTE before splitting gave {{RAD_L4_A}} on real and synthetic tumours, compared with {{RAD_NOICC_A}} for its strict reference.

Choosing the CNN epoch by test AUC added {{CNN_L5_DS}} in cross-validation but {{CNN_LOCOL5_DS}} on held-out centres (Fig. 3a, b, d), where validation AUC tracked test AUC poorly across epochs (median Spearman ρ {{CNN_VALTEST_RHO_LOCO}}, vs {{CNN_VALTEST_RHO_STRICT}} in cross-validation). Not every flaw inflated performance. ComBat fitted with the test centre lowered the LOCO AUC ({{RAD_L7H2_DS}}), although the mean per-centre AUC barely changed ({{L7H0_CEN_MEAN}} vs {{L7H2_CEN_MEAN}}): harmonisation removed between-centre differences in mean predicted risk ({{L7H0_MEANP_RANGE}} vs {{L7H2_MEANP_RANGE}}) that had broadly followed prevalence. With the label as covariate, ComBat changed the pooled cross-validation AUC by {{RAD_L7S1H3_D}} but inverted all predictions in held-out centre {{L7H3_MIN_CENTRE}} (AUC {{L7H3_MIN_AUC}}; Supplementary Methods S6). AdaBN changed the CNN AUC by {{CNN_L7_DS}}. A threshold chosen on the test data raised balanced accuracy from {{L6_BALANCED_ACCURACY_TRAIN}} to {{L6_BALANCED_ACCURACY_TEST}}.

## Centre information in image features

Radiomics features identified the acquiring centre with a macro AUC of {{ID_RAD}} (accuracy {{ID_RAD_ACC}}; chance 0.25), mainly through first-order intensity features ({{ID_FO}}); frozen deep features reached {{ID_BCLIP}}–{{ID_DINO}}, and tumour size {{ID_SIZE}} (Fig. 4a–c). In pooled cross-validation, the CENTRE model, which uses no image information, reached an AUC of {{A_S1_CENTRE}}, whereas within centres it carried no information (weighted within-centre AUC {{WC_S1_CENTRE}}; Fig. 4d).

## What the models use

The final radiomics model retained {{N_FINAL_FEAT}} features, {{N_FINAL_PT}} of them peritumoural; the most influential were {{TOP_FEAT_1}}, {{TOP_FEAT_2}} and {{TOP_FEAT_3}} (Fig. 5a). Selection was unstable: {{N_FREQ_ANY}} different features were selected across the 25 S1 folds, and only {{N_FREQ50}} in at least half of them (Fig. 5b). Grad-CAM energy of the CNN was only modestly concentrated in the tumour (enrichment {{ENRICH_T}}) and ring ({{ENRICH_R}}; Fig. 5c, d). Blurring the ring barely changed the CNN AUC ({{ABL_RING_A}} vs {{ABL_FULL_A}}), blurring the tumour reduced it to {{ABL_TUMOUR_A}}, and blurring everything outside the tumour reduced it to {{ABL_OUTSIDE_A}} (Fig. 5e).
