"""13 - Manuscript tables (CSV, rendered into the manuscript by manuscript/build.py).

Usage: python 13_tables.py [table1 table2 ...]   (default: all)
Outputs: results/tables/Table1_cohort.csv, Table2_strict.csv, Table3_audit.csv and supplementary tables TableS*.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from bcmri import config as C

T = C.TABLES


def cohort():
    coh = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    coh["y"] = coh["label"].astype(int)
    return coh


def med_iqr(v, d=1):
    q1, m, q3 = np.percentile(v, [25, 50, 75])
    return f"{m:.{d}f} ({q1:.{d}f}–{q3:.{d}f})"


def med_rng(v, d=2):
    return f"{np.median(v):.{d}f} ({np.min(v):.{d}f}–{np.max(v):.{d}f})"


def table1():
    """Per-centre characteristics. Rows marked with a dagger come from the dataset publication
    (data/external/fedbca_publication_table1_2.csv); all other rows are computed from the released files."""
    coh = cohort()
    scans = pd.read_csv(C.PROCESSED / "scan_stats.csv")
    scanner = {"C1": "MAGNETOM Skyra (Siemens)", "C2": "uMR 780 (United Imaging Healthcare)",
               "C3": "Discovery MR750w (GE Healthcare)", "C4": "MAGNETOM Verio (Siemens)"}
    pub = pd.read_csv(C.ROOT / "data" / "external" / "fedbca_publication_table1_2.csv").set_index("characteristic")
    acq_pub = [r for r in pub.index if r.startswith(("TR", "Slice thickness"))]
    pat_pub = [r for r in pub.index if r not in acq_pub]
    cols = {}
    for c in C.CENTRES + ["All"]:
        g = coh if c == "All" else coh[coh["centre"] == c]
        s_ = scans if c == "All" else scans[scans["centre"] == c]
        tpp = g.groupby("patient_id").size()
        inpl = np.sqrt(s_["sx"].values * s_["sy"].values)
        col = {"Scanner (3.0 T)": scanner.get(c, "4 scanners, 3 vendors")}
        col.update({r: pub.loc[r, c] for r in acq_pub})
        col.update({
            "In-plane pixel spacing of released images, mm, median (range)": med_rng(inpl),
            "Slice spacing of released images, mm, median (range)": med_rng(s_["sz"].values, 1),
            "Patients, n": f"{g['patient_id'].nunique()}",
            "Tumours, n": f"{len(g)}",
            "Patients with > 1 tumour, n": f"{int((tpp > 1).sum())}"})
        col.update({r: pub.loc[r, c] for r in pat_pub})
        col.update({
            "MIBC, n (%)": f"{int(g['y'].sum())} ({100 * g['y'].mean():.1f})",
            "Tumour volume, NMIBC, ml, median (IQR)": med_iqr(g.loc[g["y"] == 0, "tumour_ml"]),
            "Tumour volume, MIBC, ml, median (IQR)": med_iqr(g.loc[g["y"] == 1, "tumour_ml"]),
            "Tumour-bearing slices, median (range)": med_rng(g["n_slices_tumour"].values, 0)})
        cols[c] = col
    t = pd.DataFrame(cols)
    t.index.name = "Characteristic"
    t.to_csv(T / "Table1_cohort.csv")
    print(t.to_string())


NAMES = {("RAD_ITPT", "LR"): "Radiomics (IT+PT)", ("RAD_IT", "LR"): "Radiomics (IT only)",
         ("CNN", "ResNet18"): "Fine-tuned CNN", ("DLR", "LR"): "Deep-radiomics fusion",
         ("SIZE", "LR"): "Tumour size", ("DL_ResNet50_IN", "LR"): "ResNet-50 probe",
         ("DL_DINOv2_B", "LR"): "DINOv2 probe", ("DL_BiomedCLIP", "LR"): "BiomedCLIP probe",
         ("LATE", "mean"): "Late fusion", ("CENTRE", "prevalence"): "Centre prevalence",
         ("CNN_AdaBN", "ResNet18"): "Fine-tuned CNN + AdaBN"}
PRIMARY = [("RAD_ITPT", "LR"), ("CNN", "ResNet18"), ("DLR", "LR"), ("SIZE", "LR")]
PROTO = {"S1": "S1 Patient-grouped CV (internal)", "S2": "S2 Leave-one-centre-out (external; primary)",
         "S3": "S3 Developed on C1, tested on C2–C4"}


def c3(a, lo, hi):
    return "–" if pd.isna(a) else f"{a:.2f} ({lo:.2f}–{hi:.2f})"


def zero_sign(t):
    """'+0.00'/'-0.00' -> '0.00'; remaining hyphen-minus -> true minus sign."""
    import re
    return re.sub(r"[+-]0\.00(?!\d)", "0.00", t).replace("-", "−")


def d3(d, lo, hi):
    """Signed difference with CI, true minus signs and 'to' (as in the text): +0.18 (+0.14 to +0.23)."""
    if pd.isna(d):
        return "–"
    return zero_sign(f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})")


def name_of(m, clf):
    if (m, clf) in NAMES:
        return NAMES[(m, clf)]
    base = {"RAD_ITPT": "Radiomics (IT+PT)", "RAD_IT": "Radiomics (IT only)", "DLR": "Deep-radiomics fusion"}
    return f"{base.get(m, m)}, {clf}"


def table2():
    P = pd.read_csv(T / "perf_strict.csv")
    CMP = pd.read_csv(T / "compare_vs_size.csv")
    RE = pd.read_csv(T / "perf_random_effects.csv")
    rows = []
    for proto in ("S1", "S2", "S3"):
        for key in PRIMARY:
            r = P[(P["protocol"] == proto) & (P["model"] == key[0]) & (P["clf"] == key[1])]
            if r.empty:
                continue
            r = r.iloc[0]
            cm = CMP[(CMP["protocol"] == proto) & (CMP["model"] == key[0]) & (CMP["clf"] == key[1])]
            re = RE[(RE["model"] == key[0]) & (RE["clf"] == key[1])] if proto == "S2" else pd.DataFrame()
            rows.append({
                "Protocol": PROTO[proto], "Model": NAMES[key] + (" ‡" if key[0] == "CNN" and proto == "S1" else ""),
                "AUC (95% CI)": c3(r.auc, r.auc_lo, r.auc_hi),
                "ΔAUC vs tumour size (95% CI)": "Reference" if key[0] == "SIZE" else
                (d3(cm.iloc[0].delta_auc, cm.iloc[0].lo, cm.iloc[0].hi) if not cm.empty else "–"),
                "Pooled per-centre AUC (95% CI) [95% PI]": "" if re.empty else
                f"{c3(re.iloc[0].auc, re.iloc[0].lo, re.iloc[0].hi)} [{re.iloc[0].pi_lo:.2f}–{re.iloc[0].pi_hi:.2f}]",
                "Sensitivity": "–" if pd.isna(r.get("sensitivity")) else f"{r.sensitivity:.2f}",
                "Specificity": "–" if pd.isna(r.get("specificity")) else f"{r.specificity:.2f}",
                "Brier score": f"{r.brier:.3f}",
                "Calibration slope": f"{r.cal_slope:.2f}"})
    t = pd.DataFrame(rows)
    t.to_csv(T / "Table2_strict.csv", index=False)
    print(t.to_string())


AUDIT_DEF = [
    ("Radiomics", "Strict (patient-grouped CV)", "Reference: all fitting inside patient-grouped training folds"),
    ("Radiomics", "L2 Tumour-level split", "Tumours of one patient may be split across training and test"),
    ("Radiomics", "L3 Feature selection on all data", "Feature selection run once on all tumours before CV"),
    ("Radiomics", "Strict, no ICC filter", "Reference for L4 and L7 (no segmentation-robustness filter)"),
    ("Radiomics", "L4 SMOTE before split", "Synthetic MIBC tumours created before splitting (unpaired)"),
    ("Radiomics", "Strict, repeats 0-1", "Reference for L5 (pre-specified model, first two repeats)"),
    ("Radiomics", "L5 Best of", "Best of all classifier, hyper-parameter and feature-number configurations by test AUC"),
    ("Radiomics", "L7 ComBat on all data, label-free", "ComBat fitted on all tumours (transductive)"),
    ("Radiomics", "L7 ComBat on all data, label as covariate", "ComBat fitted on all tumours with the label as covariate"),
    ("Radiomics", "LOCO, no harmonisation", "Reference: leave-one-centre-out, no harmonisation"),
    ("Radiomics", "LOCO, ComBat on training centres only", "ComBat fitted on training centres only (deployable)"),
    ("Radiomics", "L7 LOCO, ComBat incl. test centre, label-free", "ComBat fitted including the unlabelled test centre"),
    ("Radiomics", "L7 LOCO, ComBat incl. test centre, label as covariate", "ComBat including the test centre and its labels"),
    ("CNN", "Strict (patient-grouped CV), tumour level", "Reference: patient-grouped CV, tumour-level score"),
    ("CNN", "L2 Tumour-level split", "Tumours of one patient may be split across training and test"),
    ("CNN", "Strict (patient-grouped CV), slice level", "Same strict predictions scored per slice (reference for L1)"),
    ("CNN", "L1 Slice-level random split", "Slices split at random, scored per slice"),
    ("CNN", "Epoch chosen on validation", "Reference for L5: fold-mean test AUC at the validation-chosen epoch"),
    ("CNN", "L5 Epoch chosen on test", "Epoch picked by test AUC (fold mean)"),
    ("CNN", "L1 Slice-level split, epoch chosen on validation", "L1, fold-mean test AUC (reference for L1+L5)"),
    ("CNN", "L1+L5 Slice-level split", "L1 with the epoch picked by test AUC (fold mean)"),
    ("CNN", "LOCO, standard", "Reference: leave-one-centre-out"),
    ("CNN", "L7 LOCO with AdaBN", "Batch-norm statistics re-estimated on the unlabelled test centre"),
    ("CNN", "LOCO, epoch chosen on validation", "Reference for L5 in LOCO: centre-mean test AUC at the validation-chosen epoch"),
    ("CNN", "L5 LOCO, epoch chosen on held-out centre", "Epoch picked by the AUC in the held-out centre (centre mean)"),
]
FOLD_MEAN_PAIRS = {"L5 Epoch chosen on test": "Epoch chosen on validation",
                   "L1+L5 Slice-level split": "L1 Slice-level split, epoch chosen on validation",
                   "L5 LOCO, epoch chosen on held-out centre": "LOCO, epoch chosen on validation"}


def fold_mean_delta(A, grp, prefix):
    """Plain difference (no CI) for leaky rows that are fold means, against their fold-mean reference."""
    if prefix not in FOLD_MEAN_PAIRS:
        return ""
    get = lambda pre: A[(A["group"] == grp) & A["variant"].astype(str).str.startswith(pre)]["auc"]
    a, b = get(prefix), get(FOLD_MEAN_PAIRS[prefix])
    return zero_sign(f"{a.iloc[0] - b.iloc[0]:+.2f} (fold means)") if len(a) and len(b) else ""


def table3():
    A = pd.read_csv(T / "audit_summary.csv")
    rows = []
    for grp, prefix, desc in AUDIT_DEF:
        r = A[(A["group"] == grp) & A["variant"].astype(str).str.startswith(prefix)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"Model": "Radiomics (IT+PT), LR" if grp == "Radiomics" else "Fine-tuned CNN",
                     "Variant": r.variant, "What changes": desc,
                     "Unit": r.get("unit", "tumour") if pd.notna(r.get("unit")) else "tumour",
                     "AUC (95% CI)": c3(r.auc, r.get("lo"), r.get("hi")) if pd.notna(r.get("lo")) else f"{r.auc:.2f}",
                     "ΔAUC vs reference (95% CI)": d3(r.get("delta"), r.get("delta_lo"), r.get("delta_hi"))
                     if pd.notna(r.get("delta")) else fold_mean_delta(A, grp, prefix)})
    t = pd.DataFrame(rows)
    t.to_csv(T / "Table3_audit.csv", index=False)
    print(t[["Variant", "AUC (95% CI)", "ΔAUC vs reference (95% CI)"]].to_string())


def software_table():
    """Supplementary Table S3: software versions (read from the running environment) and pretrained weights."""
    import json
    import platform
    import re
    from importlib import metadata
    rows = [{"Item": "Python", "Version / source": platform.python_version(), "Use": "all analyses"}]
    use = {"numpy": "numerics", "scipy": "statistics", "pandas": "tables", "scikit-learn": "models, CV, metrics",
           "SimpleITK": "image I/O, N4, resampling", "pyradiomics": "radiomics features", "torch": "deep learning",
           "torchvision": "image transforms", "timm": "backbones (ResNet, DINOv2)",
           "open_clip_torch": "BiomedCLIP encoder", "xgboost": "XGBoost classifier", "lightgbm": "LightGBM classifier",
           "imbalanced-learn": "SMOTE (audit L4 only)", "neuroCombat": "ComBat harmonisation",
           "statsmodels": "calibration models", "shap": "SHAP values", "scikit-image": "mask morphology",
           "matplotlib": "figures"}
    for pkg, u in use.items():
        v = metadata.version(pkg)
        if pkg == "pyradiomics":     # built from the release source archive, whose metadata version is not set
            url = json.loads(metadata.distribution(pkg).read_text("direct_url.json") or "{}").get("url", "")
            m = re.search(r"pyradiomics-(\d+\.\d+\.\d+)", url)
            v = f"{m.group(1)} (built from the release source archive)" if m else v
        rows.append({"Item": pkg, "Version / source": v, "Use": u})
    man = pd.read_csv(C.ROOT / "env" / "weights" / "MANIFEST.tsv", sep="\t", header=None,
                      names=["repo", "revision", "file", "bytes", "sha256"])
    man = man[~man["file"].str.endswith(".json")]
    for r in man.itertuples():
        rows.append({"Item": f"Weights: {r.repo} ({r.file})",
                     "Version / source": f"Hugging Face revision {r.revision[:10]}; SHA-256 {r.sha256}",
                     "Use": "fine-tuned CNN (ImageNet initialisation)" if "resnet18" in r.repo else "frozen feature extraction"})
    return pd.DataFrame(rows)


def hyperparameter_table():
    """Supplementary Table S4: fixed settings and search spaces, read from the code that ran the models."""
    import re
    from bcmri import ml
    zoo = ml.classifier_zoo(C.SEED)
    fixed = {"LR": ["solver", "class_weight"], "SVM": ["kernel", "class_weight"], "KNN": ["weights"],
             "RF": ["n_estimators", "class_weight"], "ExtraTrees": ["n_estimators", "class_weight"],
             "XGBoost": ["n_estimators", "subsample", "colsample_bytree"],
             "LightGBM": ["n_estimators", "subsample", "colsample_bytree", "class_weight"],
             "MLP": ["activation", "solver", "max_iter"]}
    rows = []
    for name in ml.CLASSIFIERS:
        est, grid = zoo[name]
        prm = est.get_params()
        fx = ", ".join(f"{k}={prm[k]}" for k in fixed[name])
        if name == "LR":
            fx = "L2 penalty (default), " + fx
        if name == "XGBoost":
            fx += ", scale_pos_weight=n(NMIBC)/n(MIBC) of the training data"
        n = int(np.prod([len(v) for v in grid.values()]))
        rows.append({"Component": f"Classifier: {name}" + (" (primary)" if name == "LR" else ""),
                     "Fixed settings": fx,
                     "Search space (grouped inner 3-fold CV, AUC)": "; ".join(f"{k} ∈ {{{', '.join(map(str, v))}}}"
                                                                         for k, v in grid.items()),
                     "Grid points": n})
    rows.append({"Component": "Feature selection (radiomics, fusion)",
                 "Fixed settings": "ICC(3,1) ≥ 0.75; Mann–Whitney p < 0.05 (≥ 20 kept); |r| ≤ 0.9; ≤ 10 features",
                 "Search space (grouped inner 3-fold CV, AUC)": "L1-LR C: 12 log-spaced values 0.01–10, one-standard-error rule",
                 "Grid points": 12})
    rows.append({"Component": "Linear probe (frozen deep features)",
                 "Fixed settings": "standardisation → PCA → L2-LR (class_weight=balanced)",
                 "Search space (grouped inner 3-fold CV, AUC)": "PCA components ∈ {5, 10, 20, 40}; C ∈ {0.001, 0.01, 0.1, 1}",
                 "Grid points": 16})
    src = (C.ROOT / "code" / "07_cnn.py").read_text()
    k = {m.group(1): m.group(2) for m in re.finditer(r"^([A-Z_]+) = ([0-9.]+)", src, re.M)}
    rows.append({"Component": "Fine-tuned CNN (ResNet-18, 2.5D)",
                 "Fixed settings": (f"input {k['SIZE']}×{k['SIZE']} px; ≤ {k['MAX_SLICES']} slices per tumour with ≥ "
                                    f"{float(k['MIN_AREA_FRAC']):.0%} of maximum area; batch {k['BATCH']}; ≤ {k['MAX_EPOCHS']} "
                                    f"epochs; early stopping patience {k['PATIENCE']} on validation AUC; AdamW (lr 1e-4 "
                                    "backbone, 1e-3 head, weight decay 1e-4); one-cycle schedule; weighted BCE"),
                 "Search space (grouped inner 3-fold CV, AUC)": "none (fixed a priori)", "Grid points": 1})
    return pd.DataFrame(rows)


SUPP = [  # number, file stem, title (also the sheet name in Supplementary_Tables.xlsx)
    ("S1", "TableS1_TRIPOD_AI", "TRIPOD+AI checklist"),
    ("S2", "TableS2_CLEAR", "CLEAR checklist"),
    ("S3", "TableS3_software", "Software versions and pretrained weights"),
    ("S4", "TableS4_hyperparameters", "Fixed settings and hyper-parameter search spaces"),
    ("S5", "TableS5_all_models", "Performance of all models under the strict protocols"),
    ("S6", "TableS6_by_centre", "Per-centre performance (S2 and S3)"),
    ("S7", "TableS7_random_effects", "Random-effects pooling of per-centre LOCO AUCs"),
    ("S8", "TableS8_vs_size", "AUC differences versus the tumour-size comparator"),
    ("S9", "TableS9_audit_full", "Leakage audit: all variants"),
    ("S10", "TableS10_L5_configurations", "Audit L5: all radiomics configurations"),
    ("S11", "TableS11_centre_identifiability", "Centre identifiability from image features"),
    ("S12", "TableS12_features", "Final radiomics model: selected features"),
]


def supplementary():
    """Supplementary tables S1-S12 as CSVs and one xlsx workbook (one sheet per table, in this order)."""
    ck = C.ROOT / "manuscript" / "checklists"
    pd.read_csv(ck / "TRIPOD_AI.tsv", sep="\t").to_csv(T / "TableS1_TRIPOD_AI.csv", index=False)
    pd.read_csv(ck / "CLEAR.tsv", sep="\t").to_csv(T / "TableS2_CLEAR.csv", index=False)
    software_table().to_csv(T / "TableS3_software.csv", index=False)
    hyperparameter_table().to_csv(T / "TableS4_hyperparameters.csv", index=False)
    P = pd.read_csv(T / "perf_strict.csv")
    P.insert(0, "Model", [name_of(m, c) for m, c in zip(P["model"], P["clf"])])
    keep = ["protocol", "Model", "model", "clf", "n", "n_mibc", "repeats", "auc", "auc_lo", "auc_hi",
            "auc_fold_mean", "auc_fold_sd", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy",
            "brier", "cal_intercept", "cal_slope", "auc_within_centre_weighted"]
    P[[c for c in keep if c in P.columns]].round(4).to_csv(T / "TableS5_all_models.csv", index=False)
    for src, dst in (("perf_by_centre", "TableS6_by_centre"), ("perf_random_effects", "TableS7_random_effects"),
                     ("compare_vs_size", "TableS8_vs_size")):
        D = pd.read_csv(T / f"{src}.csv")
        D.insert(0, "Model", [name_of(m, c) for m, c in zip(D["model"], D["clf"])])
        if "p_delong_unclustered" in D:    # S1 has one prediction per tumour and repeat: DeLong does not apply
            D.loc[D["protocol"] == "S1", "p_delong_unclustered"] = np.nan
        D.round(4).to_csv(T / f"{dst}.csv", index=False)
    pd.read_csv(T / "audit_summary.csv").round(4).to_csv(T / "TableS9_audit_full.csv", index=False)
    pd.read_csv(T / "audit_L5_configs.csv").round(4).to_csv(T / "TableS10_L5_configurations.csv", index=False)
    pd.read_csv(T / "domain_identifiability.csv").round(4).to_csv(T / "TableS11_centre_identifiability.csv",
                                                                  index=False)
    fq = pd.read_csv(T / "interp_selection_frequency.csv")
    fm = pd.read_csv(T / "interp_final_model.csv")
    s12 = fq.merge(fm, on="feature", how="outer")
    from scipy.stats import spearmanr          # correlation with tumour size (log volume), all tumours
    from bcmri import data as D
    coh = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    R = D.load_radiomics("orig").reindex(coh.index)
    lv = np.log(coh["tumour_ml"])
    s12["spearman_rho_log_volume"] = s12["feature"].map(lambda f: spearmanr(R[f], lv).statistic)
    s12.round(4).to_csv(T / "TableS12_features.csv", index=False)
    stale = sorted(set(T.glob("TableS*.csv")) - {T / f"{stem}.csv" for _, stem, _ in SUPP})
    for f in stale:                      # files from an older numbering scheme
        f.unlink()
    with pd.ExcelWriter(T / "Supplementary_Tables.xlsx") as xw:
        idx = pd.DataFrame([{"Table": f"Table {n}", "Title": t, "Sheet": n} for n, _, t in SUPP])
        idx.to_excel(xw, sheet_name="Contents", index=False)
        for n, stem, _ in SUPP:
            pd.read_csv(T / f"{stem}.csv").to_excel(xw, sheet_name=n, index=False)
    print("supplementary tables:", [stem for _, stem, _ in SUPP], "| removed stale:", [f.name for f in stale])


TABLES = {"table1": table1, "table2": table2, "table3": table3, "supplementary": supplementary}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(TABLES)):
        TABLES[name]()
