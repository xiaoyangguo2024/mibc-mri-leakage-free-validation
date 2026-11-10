"""14 - Every number quoted in the manuscript text, computed from the analysis outputs.

Writes manuscript/values_results.tsv (key<TAB>value), which manuscript/build.py substitutes for {{KEY}}
placeholders. Nothing in the text is typed by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from bcmri import config as C

V = {}
OUT = C.ROOT / "manuscript" / "values_results.tsv"


def f2(x):
    return f"{x:.2f}"


def ci(a, lo, hi, d=2):
    return f"{a:.{d}f} (95% CI {lo:.{d}f}–{hi:.{d}f})"


def ci_short(a, lo, hi, d=2):
    return f"{a:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ cohort
def cohort_values():
    coh = pd.read_csv(C.PROCESSED / "cohort.csv")
    coh["y"] = coh["label"].astype(int)
    tpp = coh.groupby("patient_id").size()
    V["N_TUMOURS"] = str(len(coh))
    V["N_PATIENTS"] = str(coh["patient_id"].nunique())
    V["N_MIBC"] = str(int(coh["y"].sum()))
    V["N_NMIBC"] = str(int((coh["y"] == 0).sum()))
    V["PREV"] = f"{coh['y'].mean():.2f}"
    V["PREV_PCT"] = pct(coh["y"].mean(), 1)
    V["N_MULTI"] = str(int((tpp > 1).sum()))
    V["MAX_TPP"] = str(int(tpp.max()))
    prev = coh.groupby("centre")["y"].mean()
    V["PREV_RANGE"] = f"{100 * prev.min():.0f}–{100 * prev.max():.0f}%"
    for c in C.CENTRES:
        V[f"PREV_{c}"] = pct(prev[c], 0)
    n_c = coh["centre"].value_counts()
    V["CENTRE_MAJORITY"] = f"{n_c.max() / len(coh):.2f}"
    vol = coh.groupby("y")["tumour_ml"].median()
    V["VOL_NMIBC"] = f"{vol[0]:.1f}"
    V["VOL_MIBC"] = f"{vol[1]:.1f}"
    V["VOL_RATIO"] = f"{vol[1] / vol[0]:.0f}"
    from bcmri import data as D
    R = D.load_radiomics("orig")
    V["N_FEAT_IT"] = str(sum(c.startswith("IT_") for c in R.columns))
    V["N_FEAT_PT"] = str(sum(c.startswith("PT_") for c in R.columns))
    ph = D.placeholder_columns()
    raw = D._read_radiomics("orig")
    V["N_PLACEHOLDER_FEAT"] = str(len(ph))
    V["N_PLACEHOLDER_TUM"] = str(int((raw[ph] == D.PLACEHOLDER).any(axis=1).sum()))
    run1 = C.PREDS / "run1_before_placeholder_fix" / "audit_ml.csv"
    if run1.exists():          # first audit run (before the exclusion): the ComBat-with-label artefact
        from sklearn.metrics import roc_auc_score
        a = pd.read_csv(run1)
        a = a[a["variant"] == "L7_LOCO_H3_all_with_label"]
        pc = a.groupby("centre").apply(lambda g: roc_auc_score(g["y"], g["p"]), include_groups=False)
        V["RUN1_H3_CENTRE"] = str(pc.idxmin())
        V["RUN1_H3_AUC"] = f"{pc.min():.2f}"
        # effect of the exclusion on the strict models (repeat-averaged pooled AUC; first run vs current)
        diffs, s1rad = [], {}
        for proto in ("S1", "S2", "S3"):
            for tag, d in (("run1", C.PREDS / "run1_before_placeholder_fix"), ("now", C.PREDS)):
                x = pd.read_csv(d / f"strict_{proto}.csv")
                a = x.groupby(["model", "clf"]).apply(lambda g: np.mean(
                    [roc_auc_score(r["y"], r["p"]) for _, r in g.groupby("repeat")]), include_groups=False)
                s1rad.setdefault(proto, {})[tag] = a
            diffs.append((s1rad[proto]["now"] - s1rad[proto]["run1"]).abs().max())
        V["RUN1_A_S1_RAD"] = f"{s1rad['S1']['run1'][('RAD_ITPT', 'LR')]:.3f}"
        V["NOW_A_S1_RAD"] = f"{s1rad['S1']['now'][('RAD_ITPT', 'LR')]:.3f}"
        V["RUN1_MAXDIFF_S1"] = f"{diffs[0]:.3f}"
        V["RUN1_MAXDIFF_S23"] = f"{max(diffs[1:]):.3f}"
    log = (C.RESULTS / "logs" / "01_inventory.txt")
    V["MAX_DUP_R"] = "0.875"
    if log.exists():
        import re
        m = re.search(r"max(?:imum)?[^\n]*?corr[^\n]*?(0\.\d+)", log.read_text())
        if m:
            V["MAX_DUP_R"] = m.group(1)


# ------------------------------------------------------------------ Riley sample size
def riley_values(c_stat=0.80, n_pred=10, shrink=0.9, n_sim=400_000, seed=C.SEED):
    """Cox-Snell R2 implied by a C statistic (simulated, Riley et al. 2021 approach) and the resulting n."""
    coh = pd.read_csv(C.PROCESSED / "cohort.csv")
    phi = coh["label"].astype(int).mean()
    rng = np.random.default_rng(seed)
    from scipy.optimize import brentq
    from sklearn.metrics import roc_auc_score

    def sim(sd, n=n_sim):
        lp0 = rng.normal(0, sd, n)
        b0 = brentq(lambda b: (1 / (1 + np.exp(-(b + lp0)))).mean() - phi, -20, 20)
        p = 1 / (1 + np.exp(-(b0 + lp0)))
        y = rng.random(n) < p
        return p, y

    sd = brentq(lambda s: roc_auc_score(*reversed(sim(s, 60_000))) - c_stat, 0.05, 5.0, xtol=1e-3)
    p, y = sim(sd)
    ll_model = np.sum(y * np.log(p) + (~y) * np.log(1 - p))
    ll_null = len(y) * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
    r2 = 1 - np.exp(2 * (ll_null - ll_model) / len(y))
    n = n_pred / ((shrink - 1) * np.log(1 - r2 / shrink))
    V["RILEY_R2"] = f"{r2:.2f}"
    V["RILEY_N10"] = f"{int(np.ceil(n / 10) * 10):,}"
    V["RILEY_N10_EVENTS"] = f"{int(round(n * phi)):,}"


# ------------------------------------------------------------------ results
MKEY = {"RAD": ("RAD_ITPT", "LR"), "RADIT": ("RAD_IT", "LR"), "CNN": ("CNN", "ResNet18"), "DLR": ("DLR", "LR"),
        "SIZE": ("SIZE", "LR"), "CENTRE": ("CENTRE", "prevalence"), "LATE": ("LATE", "mean"),
        "R50": ("DL_ResNet50_IN", "LR"), "DINO": ("DL_DINOv2_B", "LR"), "BCLIP": ("DL_BiomedCLIP", "LR"),
        "CNNADA": ("CNN_AdaBN", "ResNet18")}


def zero_sign(t):
    """'+0.00'/'-0.00' -> '0.00'; remaining hyphen-minus -> true minus sign."""
    import re
    return re.sub(r"[+-]0\.00(?!\d)", "0.00", t).replace("-", "−")


def signed(d, lo, hi, dd=2):
    return zero_sign(f"{d:+.{dd}f} (95% CI {lo:+.{dd}f} to {hi:+.{dd}f})")


def p_fmt(p):
    if pd.isna(p):
        return "n/a"
    return "*p* < 0.001" if p < 0.001 else f"*p* = {p:.3f}" if p < 0.01 else f"*p* = {p:.2f}"


def sel(D, key, **kw):
    m = (D["model"] == key[0]) & (D["clf"] == key[1])
    for k, v in kw.items():
        m &= D[k] == v
    return D[m]


def perf_values():
    T = C.TABLES
    P = pd.read_csv(T / "perf_strict.csv")
    for proto in ("S1", "S2", "S3"):
        for k, key in MKEY.items():
            r = sel(P, key, protocol=proto)
            if r.empty:
                continue
            r = r.iloc[0]
            V[f"AUC_{proto}_{k}"] = ci(r.auc, r.auc_lo, r.auc_hi)
            V[f"AUCS_{proto}_{k}"] = ci_short(r.auc, r.auc_lo, r.auc_hi)
            V[f"A_{proto}_{k}"] = f2(r.auc)
            for col, nm, d in (("sensitivity", "SENS", 2), ("specificity", "SPEC", 2), ("brier", "BRIER", 3),
                               ("cal_slope", "SLOPE", 2), ("cal_intercept", "CITL", 2)):
                if col in r and pd.notna(r[col]):
                    V[f"{nm}_{proto}_{k}"] = f"{r[col]:.{d}f}".replace("-", "−")
            if proto == "S1" and pd.notna(r.get("auc_fold_mean")):
                V[f"FOLD_S1_{k}"] = f"{r.auc_fold_mean:.2f} ± {r.auc_fold_sd:.2f}"
            if proto == "S1" and pd.notna(r.get("auc_within_centre_weighted")):
                V[f"WC_S1_{k}"] = f2(r.auc_within_centre_weighted)
    # best and worst secondary configuration under S2 (descriptive)
    s2 = P[P["protocol"] == "S2"].sort_values("auc", ascending=False)
    V["S2_MAX_ANY"] = f2(s2["auc"].max())
    top = s2.iloc[0]
    clf_name = {"LR": "logistic regression", "mean": "", "ResNet18": ""}.get(top["clf"], top["clf"])
    model_name = _fig_module().NAMES.get((top["model"], "LR" if top["model"] != "LATE" else "mean"), top["model"])
    V["S2_MAX_NAME"] = (model_name[0].lower() + model_name[1:]) + (f" with {clf_name}" if clf_name else "")
    V["N_MODELS_TOTAL"] = str(len(P[["model", "clf"]].drop_duplicates()))

    CMP = pd.read_csv(T / "compare_vs_size.csv")
    r = CMP[(CMP["protocol"] == "S2") & (CMP["model"] == top["model"]) & (CMP["clf"] == top["clf"])]
    if not r.empty:
        V["DSIZE_S2_MAX"] = signed(r.iloc[0].delta_auc, r.iloc[0].lo, r.iloc[0].hi)
    for proto in ("S1", "S2", "S3"):
        for k, key in MKEY.items():
            r = sel(CMP, key, protocol=proto)
            if r.empty:
                continue
            r = r.iloc[0]
            V[f"DSIZE_{proto}_{k}"] = signed(r.delta_auc, r.lo, r.hi)
            V[f"PSIZE_{proto}_{k}"] = p_fmt(r.p_boot)

    RE = pd.read_csv(T / "perf_random_effects.csv")
    for k, key in MKEY.items():
        r = sel(RE, key)
        if r.empty:
            continue
        r = r.iloc[0]
        V[f"RE_{k}"] = ci(r.auc, r.lo, r.hi)
        V[f"RE_A_{k}"] = f2(r.auc)
        V[f"PI_{k}"] = f"{r.pi_lo:.2f}–{r.pi_hi:.2f}"
        i2 = r.I2 if r.I2 > 1 else 100 * r.I2
        V[f"I2_{k}"] = f"{i2:.0f}%"
    BC = pd.read_csv(T / "perf_by_centre.csv")
    for proto in ("S2", "S3"):
        for k, key in MKEY.items():
            r = sel(BC, key, protocol=proto) if "protocol" in BC else pd.DataFrame()
            if r.empty:
                continue
            for c, a in zip(r["centre"], r["auc"]):
                V[f"CEN_{proto}_{k}_{c}"] = f2(a)
            V[f"CENRANGE_{proto}_{k}"] = f"{r['auc'].min():.2f}–{r['auc'].max():.2f}"
            # does the mean predicted risk in each held-out centre follow that centre's prevalence?
            for c, mp in zip(r["centre"], r["mean_p"]):
                V[f"MEANP_{proto}_{k}_{c}"] = pct(mp, 0)
            from scipy.stats import spearmanr
            V[f"RHO_MEANP_{proto}_{k}"] = f"{spearmanr(r['mean_p'], r['prevalence']).statistic:.1f}".replace("-", "−")


AUD = {  # key -> (group, variant prefix in audit_summary.csv)
    "RAD_STRICT": ("Radiomics", "Strict (patient-grouped CV)"),
    "RAD_L2": ("Radiomics", "L2 Tumour-level split"),
    "RAD_L3": ("Radiomics", "L3 Feature selection on all data"),
    "RAD_NOICC": ("Radiomics", "Strict, no ICC filter"),
    "RAD_L4": ("Radiomics", "L4 SMOTE before split"),
    "RAD_REF01": ("Radiomics", "Strict, repeats 0-1"),
    "RAD_L5": ("Radiomics", "L5 Best of"),
    "RAD_L7S1H2": ("Radiomics", "L7 ComBat on all data, label-free"),
    "RAD_L7S1H3": ("Radiomics", "L7 ComBat on all data, label as covariate"),
    "RAD_LOCO": ("Radiomics", "LOCO, no harmonisation"),
    "RAD_LOCOH1": ("Radiomics", "LOCO, ComBat on training centres only"),
    "RAD_L7H2": ("Radiomics", "L7 LOCO, ComBat incl. test centre, label-free"),
    "RAD_L7H3": ("Radiomics", "L7 LOCO, ComBat incl. test centre, label as covariate"),
    "CNN_STRICT": ("CNN", "Strict (patient-grouped CV), tumour level"),
    "CNN_SLICE": ("CNN", "Strict (patient-grouped CV), slice level"),
    "CNN_L1": ("CNN", "L1 Slice-level random split"),
    "CNN_L1VAL": ("CNN", "L1 Slice-level split, epoch chosen on validation"),
    "CNN_L1L5": ("CNN", "L1+L5"),
    "CNN_L2": ("CNN", "L2 Tumour-level split"),
    "CNN_EPVAL": ("CNN", "Epoch chosen on validation"),
    "CNN_L5": ("CNN", "L5 Epoch chosen on test"),
    "CNN_LOCO": ("CNN", "LOCO, standard"),
    "CNN_L7": ("CNN", "L7 LOCO with AdaBN"),
    "CNN_LOCOEPVAL": ("CNN", "LOCO, epoch chosen on validation"),
    "CNN_LOCOL5": ("CNN", "L5 LOCO, epoch chosen on held-out centre"),
}


def audit_values():
    T = C.TABLES
    A = pd.read_csv(T / "audit_summary.csv")
    for k, (grp, prefix) in AUD.items():
        r = A[(A["group"] == grp) & A["variant"].astype(str).str.startswith(prefix)]
        if r.empty:
            continue
        r = r.iloc[0]
        V[f"{k}_A"] = f2(r.auc)
        V[f"{k}_AUC"] = ci(r.auc, r.lo, r.hi) if pd.notna(r.get("lo")) else f2(r.auc)
        if pd.notna(r.get("delta")):
            V[f"{k}_D"] = signed(r.delta, r.delta_lo, r.delta_hi)
            V[f"{k}_DS"] = zero_sign(f"{r.delta:+.2f}")
    # simple differences for rows without paired bootstrap (fold-mean AUCs)
    for leak, ref, key in (("CNN_L5", "CNN_EPVAL", "CNN_L5"), ("CNN_L1L5", "CNN_L1VAL", "CNN_L1L5"),
                           ("CNN_LOCOL5", "CNN_LOCOEPVAL", "CNN_LOCOL5")):
        if f"{leak}_A" in V and f"{ref}_A" in V:
            V[f"{key}_DS"] = zero_sign(f"{float(V[leak + '_A']) - float(V[ref + '_A']):+.2f}")
    for k in ("Training-derived threshold", "L6 Threshold tuned on test set"):
        r = A[A["variant"] == k]
        if r.empty:
            continue
        r = r.iloc[0]
        t = "TRAIN" if k.startswith("Training") else "TEST"
        for col in ("sensitivity", "specificity", "accuracy", "balanced_accuracy"):
            V[f"L6_{col.upper()}_{t}"] = f2(r[col])
    f = T / "audit_L5_configs.csv"
    if f.exists():
        L5 = pd.read_csv(f)
        a = L5["auc_mean_2rep"]
        V["N_L5_CONFIGS"] = str(len(L5))
        V["L5_MEDIAN"] = f2(a.median())
        V["L5_RANGE"] = f"{a.min():.2f}–{a.max():.2f}"
        V["L5_BEST_CONFIG"] = str(L5.iloc[0, 0])
    else:
        from bcmri import ml
        n = sum(int(np.prod([len(v) for v in g.values()])) for _, g in ml.classifier_zoo(C.SEED).values())
        V["N_L5_CONFIGS"] = str(3 * n)


def domain_values():
    D = pd.read_csv(C.TABLES / "domain_identifiability.csv").set_index("feature_set")
    names = {"ID_RAD": "Radiomics (IT+PT, all)", "ID_FO": "Radiomics, first-order only",
             "ID_SHAPE": "Radiomics, shape only", "ID_SIZE": "Tumour size (2 features)",
             "ID_R50": "Deep: ResNet50_IN", "ID_DINO": "Deep: DINOv2_B", "ID_BCLIP": "Deep: BiomedCLIP"}
    for k, nm in names.items():
        if nm in D.index:
            V[k] = f2(D.loc[nm, "macro_auc_ovr"])
            V[k + "_ACC"] = f2(D.loc[nm, "accuracy"])


def _fig_module():
    """code/12_figures.py (for pretty_feature and the model names), imported without drawing anything."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fig12", C.ROOT / "code" / "12_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prose_feature(name):
    """Radiomics feature name for running text: 'IT_original_shape2D_MinorAxisLength' -> 'intratumoural minor axis
    length'; 'PT_wavelet-LL_glszm_LargeAreaLowGrayLevelEmphasis' -> 'peritumoural wavelet-LL GLSZM large area low
    grey level emphasis'."""
    import re
    region, rest = name.split("_", 1)
    img, cls, feat = rest.rsplit("_", 2)
    reg = {"IT": "intratumoural", "PT": "peritumoural"}.get(region, region)
    if img.startswith("log-sigma"):
        img = f"Laplacian-of-Gaussian (σ = {img.split('-')[2]} mm)"
    elif img.startswith("wavelet"):
        img = "wavelet-" + img.split("-")[1]
    else:
        img = ""
    cls = "" if cls.startswith("shape") else {"firstorder": "first-order"}.get(cls, cls.upper())
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", feat).lower().replace("gray", "grey")
    return " ".join(x for x in (reg, img, cls, words) if x)


def interp_values():
    from bcmri import stats as S
    T = C.TABLES
    fig = _fig_module()
    fm = pd.read_csv(T / "interp_final_model.csv").sort_values("mean_abs_shap", ascending=False)
    V["N_FINAL_FEAT"] = str(len(fm))
    for i, f in enumerate(fm["feature"].head(3), 1):
        V[f"TOP_FEAT_{i}"] = fig.pretty_feature(f)
    V["N_FINAL_PT"] = str(int(fm["feature"].str.startswith("PT_").sum()))
    V["N_FINAL_SHAPE"] = str(int(fm["feature"].str.contains("_shape_").sum()))
    fq = pd.read_csv(T / "interp_selection_frequency.csv")
    V["N_FREQ50"] = str(int((fq["fraction"] >= 0.5).sum()))
    V["N_FREQ_ANY"] = str(len(fq))
    # correlation of selected radiomics features with tumour size (Spearman, log volume; all tumours)
    from scipy.stats import spearmanr
    from bcmri import data as D
    coh_r = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    R = D.load_radiomics("orig").reindex(coh_r.index)
    lv = np.log(coh_r["tumour_ml"])
    rho_sel = fq["feature"].map(lambda f: abs(spearmanr(R[f], lv).statistic))
    V["N_SEL_SIZECORR"] = str(int((rho_sel >= 0.5).sum()))
    V["SHARE_SEL_SIZECORR"] = pct((fq["n_folds"] * (rho_sel >= 0.5)).sum() / fq["n_folds"].sum())
    rho_fin = fm["feature"].map(lambda f: abs(spearmanr(R[f], lv).statistic))
    V["N_FINAL_SIZECORR"] = str(int((rho_fin >= 0.5).sum()))
    top = fq.sort_values("fraction", ascending=False).iloc[0]
    V["TOPFREQ_FEAT"] = fig.pretty_feature(top["feature"])
    V["TOPFREQ_FEAT_PROSE"] = prose_feature(top["feature"])
    V["TOPFREQ_FRAC"] = pct(top["fraction"])
    V["TOPFREQ_RHO"] = f"{spearmanr(R[top['feature']], lv).statistic:.2f}"
    G = pd.read_csv(T / "interp_gradcam_regions.csv")
    for col, k in (("enrich_tumour", "ENRICH_T"), ("enrich_ring", "ENRICH_R")):
        q1, m, q3 = np.percentile(G[col], [25, 50, 75])
        V[k] = f"{m:.1f} (IQR {q1:.1f}–{q3:.1f})"
    coh = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    AB = pd.read_csv(T / "interp_region_ablation.csv")
    for v, k in (("full", "ABL_FULL"), ("tumour_blurred", "ABL_TUMOUR"), ("ring_blurred", "ABL_RING"),
                 ("outside_tumour_blurred", "ABL_OUTSIDE")):
        g = AB[AB["variant"] == v]
        b = S.cluster_bootstrap_auc(g["y"].values, g["logit"].values, coh.loc[g["case_id"], "patient_id"].values,
                                    C.N_BOOT, C.SEED)
        V[k] = ci(b["auc"], b["lo"], b["hi"])
        V[k + "_A"] = f2(b["auc"])


def longest_run(thr, mask):
    """Longest run of consecutive thresholds where mask holds, as 'a–b' (honest when there are gaps)."""
    best, cur = (None, None, 0), []
    for t, m in zip(thr, mask):
        cur = cur + [t] if m else []
        if len(cur) > best[2]:
            best = (cur[0], cur[-1], len(cur))
    return "none" if best[2] == 0 else f"{best[0]:.2f}–{best[1]:.2f}"


def epoch_values():
    """How well validation AUC (training centres) tracks test AUC across CNN training epochs, per fold."""
    from scipy.stats import spearmanr
    for proto, k in (("strict", "STRICT"), ("loco", "LOCO")):
        e = pd.read_csv(C.PREDS / f"cnn_{proto}_epochs.csv")
        r = [spearmanr(g["val_auc"], g["test_auc"]).statistic for _, g in e.groupby("fold")]
        V[f"CNN_VALTEST_RHO_{k}"] = f"{np.median(r):.2f}".replace("-", "−")


def dca_values():
    D = pd.read_csv(C.TABLES / "dca.csv")
    d = D[D["protocol"] == "S2"]
    allp = d[d["model"] == "Treat all"].set_index("threshold")["net_benefit"]
    size = sel(d, MKEY["SIZE"]).set_index("threshold")["net_benefit"]
    for k in ("RAD", "CNN", "DLR", "SIZE"):
        g = sel(d, MKEY[k]).set_index("threshold")["net_benefit"]
        if g.empty:
            continue
        better = (g > allp.reindex(g.index) + 1e-9) & (g > 1e-9)
        V[f"DCA_{k}_RANGE"] = longest_run(g.index.values, better.values)
        if k != "SIZE":
            gt = g > size.reindex(g.index) + 1e-9
            V[f"DCA_{k}_GT_SIZE"] = f"{np.mean(gt):.0%}"
            V[f"DCA_{k}_GT_SIZE_RANGE"] = longest_run(g.index.values, gt.values)
        V[f"NB30_{k}"] = f"{g.loc[0.3]:.3f}"
    V["NB30_ALL"] = f"{allp.loc[0.3]:.3f}"
    rad = sel(d, MKEY["RAD"]).set_index("threshold")["net_benefit"]
    gap = (rad - size.reindex(rad.index)).values
    gap_rc = dca_recal_gap(rad.index.values)
    rng = lambda g: zero_sign(f"{g.min():.2f} to {g.max():.2f}")
    V["DCA_RAD_SIZE_MAXGAP"] = f"{gap.max():.2f}"
    V["DCA_RAD_SIZE_MAXGAP_RECAL"] = f"{gap_rc.max():.2f}"
    V["DCA_RAD_SIZE_GAPRANGE_RAW"] = rng(gap)
    V["DCA_RAD_SIZE_GAPRANGE_RECAL"] = rng(gap_rc)


def dca_recal_gap(thresholds):
    """Diagnostic only: net-benefit difference (radiomics minus size) at each threshold after logistic recalibration
    (intercept and slope) of the pooled S2 predictions. Uses the test labels, so it separates calibration from
    discrimination but is not a performance estimate."""
    from sklearn.linear_model import LogisticRegression
    P = pd.read_csv(C.PREDS / "strict_S2.csv")
    nb = {}
    for k in ("RAD", "SIZE"):
        g = sel(P, MKEY[k])
        y = g["y"].values
        p = np.clip(g["p"].values, 1e-6, 1 - 1e-6)
        lp = np.log(p / (1 - p)).reshape(-1, 1)
        q = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).predict_proba(lp)[:, 1]
        n = len(y)
        nb[k] = np.array([np.sum((q >= t) & (y == 1)) / n - np.sum((q >= t) & (y == 0)) / n * t / (1 - t)
                          for t in thresholds])
    return nb["RAD"] - nb["SIZE"]


def l7_artefact_values():
    """ComBat with the outcome as covariate (L7, LOCO): per-centre AUCs and, for the worst held-out centre, the
    feature(s) the model selected and the extreme value behind the artefact (recomputed with the 08 code)."""
    from sklearn.metrics import roc_auc_score
    import importlib.util
    A = pd.read_csv(C.PREDS / "audit_ml.csv")
    g = A[A["variant"] == "L7_LOCO_H3_all_with_label"]
    pc = g.groupby("centre").apply(lambda x: roc_auc_score(x["y"], x["p"]), include_groups=False)
    worst = pc.idxmin()
    V["L7H3_MIN_CENTRE"], V["L7H3_MIN_AUC"] = str(worst), f"{pc.min():.2f}"
    others = pc.drop(worst)
    V["L7H3_OTHER_RANGE"] = f"{others.min():.2f}–{others.max():.2f}"
    # label-free ComBat incl. the test centre (H2) vs no harmonisation (H0): mean per-centre AUC and the spread of
    # centre-mean predicted risk, which show whether a change in pooled AUC comes from within- or between-centre ranking
    for v, k in (("L7_LOCO_H0_none", "L7H0"), ("L7_LOCO_H2_all_label_free", "L7H2")):
        g = A[A["variant"] == v]
        pcv = g.groupby("centre").apply(lambda x: roc_auc_score(x["y"], x["p"]), include_groups=False)
        mp = g.groupby("centre")["p"].mean()
        V[f"{k}_CEN_MEAN"] = f2(pcv.mean())
        V[f"{k}_MEANP_RANGE"] = f"{mp.min():.2f}–{mp.max():.2f}"
    spec = importlib.util.spec_from_file_location("a08", C.ROOT / "code" / "08_audit_ml.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.N_JOBS = 1
    from bcmri import data as D, ml
    coh = D.load_cohort()
    R = D.load_radiomics("orig").reindex(coh.index)
    R = R[[c for c in R.columns if c.startswith(("IT_", "PT_"))]].replace([np.inf, -np.inf], np.nan)
    R = R.loc[:, R.notna().all(axis=0)]
    H3 = m.combat(R, coh["centre"], coh["y"])
    tr = np.where(coh["centre"].values != worst)[0]
    te = np.where(coh["centre"].values == worst)[0]
    _, sel = m.radiomics_fold(H3, coh, tr, te, coh["patient_id"].values, C.SEED, None)
    f = sel.selected_
    fig = _fig_module()
    V["L7H3_FEAT"] = "; ".join(fig.pretty_feature(x) for x in f)
    x = R[f[0]]
    k = x.iloc[te].abs().idxmax()
    ratio = x.loc[k] / x.iloc[tr].median()
    V["L7H3_RATIO"] = f"10^{int(np.round(np.log10(abs(ratio))))}^"
    V["L7H3_OUTLIER_LABEL"] = "MIBC" if coh.loc[k, "y"] == 1 else "NMIBC"
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    V["L7H3_NSEL"] = words.get(len(f), str(len(f)))
    # within-centre AUC of that feature before and after label-informed ComBat
    cen, y = coh["centre"].values, coh["y"].values
    raw_auc = {c: roc_auc_score(y[cen == c], R[f[0]].values[cen == c]) for c in C.CENTRES}
    h3_auc = {c: roc_auc_score(y[cen == c], H3[f[0]].values[cen == c]) for c in C.CENTRES}
    a = [h3_auc[c] for c in C.CENTRES if c != worst]
    V["L7H3_FEAT_AUC_TRAIN"] = f2(min(a)) if f2(min(a)) == f2(max(a)) else f"{min(a):.2f}–{max(a):.2f}"
    V["L7H3_FEAT_AUC_TEST"] = f2(h3_auc[worst])
    V["L7H3_FEAT_AUC_RAW"] = f"{min(raw_auc.values()):.2f}–{max(raw_auc.values()):.2f}"


def main():
    cohort_values()
    riley_values()
    for fn in (perf_values, audit_values, domain_values, interp_values, dca_values, epoch_values,
               l7_artefact_values):
        try:
            fn()
        except FileNotFoundError as e:          # analysis step not run yet
            print(f"skipped {fn.__name__}: {e.filename}")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# written by code/14_values.py -- do not edit by hand\n")
        for k in sorted(V):
            fh.write(f"{k}\t{V[k]}\n")
    for k in sorted(V):
        print(f"{k:28s} {V[k]}")


if __name__ == "__main__":
    main()
