# Abstract

**Objectives** To determine how much of the reported performance of MRI radiomics and deep learning for detecting muscle invasion in bladder cancer survives leakage-free multicentre validation, and how much common evaluation flaws inflate it.

**Materials and methods** This retrospective study used the public four-centre FedBCa T2-weighted MRI dataset ({{N_TUMOURS}} tumours, {{N_PATIENTS}} patients, {{N_MIBC}} muscle-invasive). Pre-specified radiomics, fine-tuned convolutional neural network (CNN) and deep-radiomics fusion models were compared with tumour size under patient-grouped cross-validation and leave-one-centre-out validation (primary endpoint). Eight common evaluation flaws were then introduced one at a time into the same pipelines.

**Results** On held-out centres, the area under the ROC curve (AUC) was {{AUCS_S2_RAD}} for radiomics, {{AUCS_S2_DLR}} for fusion and {{AUCS_S2_CNN}} for the CNN, compared with {{AUCS_S2_SIZE}} for tumour size; no image model significantly outperformed size. Random slice-level splitting raised the CNN AUC from {{CNN_SLICE_A}} to {{CNN_L1_A}}. Selecting features on all tumours and reporting the best of {{N_L5_CONFIGS}} configurations added {{RAD_L3_DS}} and {{RAD_L5_DS}} to the radiomics AUC. Radiomics features identified the acquiring centre (macro AUC {{ID_RAD}}), and a model using only centre prevalence reached an AUC of {{A_S1_CENTRE}} in pooled cross-validation.

**Conclusion** Under leakage-free multicentre validation, T2-weighted MRI radiomics and deep learning did not detect muscle invasion better than tumour size. Single evaluation flaws, above all slice-level splitting, inflated AUCs up to the near-perfect values reported in the literature.

**Keywords** Urinary bladder neoplasms; Magnetic resonance imaging; Radiomics; Deep learning; Reproducibility of results

# Key points

**Question** Do MRI radiomics and deep-learning models for muscle-invasive bladder cancer keep their reported accuracy when evaluation leakage is removed and centres are held out?

**Findings** On unseen centres, no image model significantly exceeded tumour size; random slice splitting raised the deep-learning area under the curve from {{CNN_SLICE_A}} to {{CNN_L1_A}}.

**Critical relevance statement** Before clinical use, MRI models for muscle invasion should be tested on held-out centres, with patient-level splits, every data-dependent step confined to training data and a tumour-size comparator, because single evaluation flaws produced near-perfect but spurious accuracy.

# Abbreviations

AdaBN Adaptive batch normalisation; AUC Area under the receiver operating characteristic curve; CI Confidence interval; CNN Convolutional neural network; DLR Deep-radiomics fusion; ICC Intraclass correlation coefficient; IT Intratumoural; LOCO Leave-one-centre-out; MIBC Muscle-invasive bladder cancer; NMIBC Non-muscle-invasive bladder cancer; PT Peritumoural; ROC Receiver operating characteristic; SHAP Shapley additive explanations; SMOTE Synthetic minority over-sampling technique; T2WI T2-weighted imaging; TURBT Transurethral resection of bladder tumour; VI-RADS Vesical Imaging-Reporting and Data System
