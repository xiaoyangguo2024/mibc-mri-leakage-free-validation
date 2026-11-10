"""11 - Performance statistics for every model / protocol / audit variant.

All CIs: patient-level cluster bootstrap (2000 resamples). S1 statistics are means over the
5 repeats, evaluated on the same patient resamples. Thresholds for sensitivity / specificity
come from the training data of each split (column `thr`), never from the test data -- except in
the audit row that deliberately re-derives the Youden threshold on the test predictions (L6).
Outputs (results/tables/): perf_strict.csv, perf_by_centre.csv, perf_random_effects.csv,
compare_vs_size.csv, audit_summary.csv, audit_L5_configs.csv, dca.csv, calibration_curves.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bcmri import config as C
from bcmri import data as D
from bcmri import stats as S

NB = C.N_BOOT
KEY = [("RAD_ITPT", "LR"), ("RAD_IT", "LR"), ("DLR", "LR"), ("SIZE", "LR"), ("LATE", "mean"),
       ("DL_ResNet50_IN", "LR"), ("DL_DINOv2_B", "LR"), ("DL_BiomedCLIP", "LR"), ("CNN", "ResNet18")]


def cohort():
    coh = D.load_cohort()
    return coh[["y", "centre", "patient_id"]]


def load_strict(proto):
    d = pd.read_csv(C.PREDS / f"strict_{proto}.csv")
    return d


def load_cnn(proto, coh):
    d = pd.read_csv(C.PREDS / f"cnn_{proto}.csv")
    d = d.drop(columns=[c for c in ("y",) if c in d.columns]).join(coh, on="case_id")
    d["model"], d["clf"], d["repeat"] = "CNN", "ResNet18", 0
    d["thr"] = np.nan
    return d


def matrix(d, coh):
    """(n_cases x R) probability matrix aligned to the cohort order (NaN-free rows only)."""
    P = d.pivot_table(index="case_id", columns="repeat", values="p", aggfunc="mean")
    P = P.reindex(coh.index).dropna()
    return P


def thr_metrics(d):
    """Sensitivity etc. with training-derived thresholds; mean over repeats."""
    if d["thr"].isna().all():
        return {}
    out = []
    for r, g in d.groupby("repeat"):
        pred = (g["p"].values >= g["thr"].values).astype(int)
        m = S.threshold_metrics(g["y"].values, pred, 0.5)
        out.append(m)
    keys = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy"]
    return {k: float(np.nanmean([o[k] for o in out])) for k in keys}


def calib(d):
    res = []
    for r, g in d.groupby("repeat"):
        citl, slope = S.calibration_intercept_slope(g["y"].values, g["p"].values)
        res.append((S.brier(g["y"].values, g["p"].values), citl, slope))
    res = np.array(res)
    return {"brier": res[:, 0].mean(), "cal_intercept": res[:, 1].mean(), "cal_slope": res[:, 2].mean()}


def perf_row(d, coh, label):
    P = matrix(d, coh)
    y = coh.loc[P.index, "y"].values
    g = coh.loc[P.index, "patient_id"].values
    b = S.cluster_bootstrap_auc(y, P.values, g, NB, C.SEED)
    row = {**label, "n": len(P), "n_mibc": int(y.sum()), "repeats": P.shape[1],
           "auc": b["auc"], "auc_lo": b["lo"], "auc_hi": b["hi"]}
    fold_aucs = [roc_auc_score(gg["y"], gg["p"]) for _, gg in d.groupby(["repeat", "fold"]) if gg["y"].nunique() == 2]
    row["auc_fold_mean"], row["auc_fold_sd"] = float(np.mean(fold_aucs)), float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else np.nan
    row.update(thr_metrics(d))
    row.update(calib(d))
    return row, P


def by_centre(d, coh, label):
    rows = []
    for c in C.CENTRES:
        g = d[d["centre"] == c]
        if g.empty or g["y"].nunique() < 2:
            continue
        P = matrix(g, coh.loc[coh["centre"] == c])
        y = coh.loc[P.index, "y"].values
        b = S.cluster_bootstrap_auc(y, P.values, coh.loc[P.index, "patient_id"].values, NB, C.SEED)
        rows.append({**label, "centre": c, "n": len(P), "n_mibc": int(y.sum()), "auc": b["auc"],
                     "auc_lo": b["lo"], "auc_hi": b["hi"], "se_logit": b["se_logit"],
                     "prevalence": float(y.mean()), "mean_p": float(P.values.mean())})
    return rows


def main():
    coh = cohort()
    perf, centre_rows, re_rows, cmp_rows = [], [], [], []
    mats = {}
    sources = {}
    for proto in ("S1", "S2", "S3"):
        d = load_strict(proto)
        for (m, clf), g in d.groupby(["model", "clf"]):
            sources[(proto, m, clf)] = g
    for proto, cproto in (("S1", "strict"), ("S2", "loco"), ("S3", "s3")):
        f = C.PREDS / f"cnn_{cproto}.csv"
        if f.exists():
            g = load_cnn(cproto, coh)
            sources[(proto, "CNN", "ResNet18")] = g[g["variant"] == "standard"]
            if cproto == "loco":
                sources[(proto, "CNN_AdaBN", "ResNet18")] = g[g["variant"] == "AdaBN"]
    for (proto, m, clf), g in sources.items():
        label = {"protocol": proto, "model": m, "clf": clf}
        row, P = perf_row(g, coh, label)
        perf.append(row)
        mats[(proto, m, clf)] = P
        if proto in ("S2", "S3"):
            cr = by_centre(g, coh, label)
            centre_rows += cr
            if proto == "S2" and len(cr) >= 3:
                re = S.random_effects_auc([r["auc"] for r in cr], [r["se_logit"] for r in cr])
                re_rows.append({**label, **re})
        if proto == "S1":
            # within-centre AUC (does pooled AUC exceed what the model achieves inside each centre?)
            wc = []
            for c in C.CENTRES:
                gc = g[g["centre"] == c]
                aucs = [roc_auc_score(x["y"], x["p"]) for _, x in gc.groupby("repeat") if x["y"].nunique() == 2]
                wc.append((len(gc) / g["repeat"].nunique(), np.mean(aucs)))
            w = np.array(wc)
            perf[-1]["auc_within_centre_weighted"] = float(np.sum(w[:, 0] * w[:, 1]) / np.sum(w[:, 0]))
    P = pd.DataFrame(perf).sort_values(["protocol", "auc"], ascending=[True, False])
    P.to_csv(C.TABLES / "perf_strict.csv", index=False)
    pd.DataFrame(centre_rows).to_csv(C.TABLES / "perf_by_centre.csv", index=False)
    pd.DataFrame(re_rows).to_csv(C.TABLES / "perf_random_effects.csv", index=False)
    print(P[["protocol", "model", "clf", "auc", "auc_lo", "auc_hi", "sensitivity", "specificity", "brier", "cal_slope"]]
          .round(3).to_string(), flush=True)

    # ---- comparisons against the size baseline (paired, same tumours, same splits)
    for proto in ("S1", "S2", "S3"):
        if (proto, "SIZE", "LR") not in mats:
            continue
        ref = mats[(proto, "SIZE", "LR")]
        for key, Pm in mats.items():
            if key[0] != proto or key[1] in ("SIZE", "CENTRE"):
                continue
            ids = Pm.index.intersection(ref.index)
            cols = [c for c in Pm.columns if c in ref.columns]
            a, b = Pm.loc[ids, cols].values, ref.loc[ids, cols].values
            r = S.cluster_bootstrap_diff(coh.loc[ids, "y"].values, a, b, coh.loc[ids, "patient_id"].values, NB, C.SEED)
            dl = S.delong_test(coh.loc[ids, "y"].values, a.mean(1), b.mean(1))
            cmp_rows.append({"protocol": proto, "model": key[1], "clf": key[2], "vs": "SIZE-LR",
                             "delta_auc": r["diff"], "lo": r["lo"], "hi": r["hi"], "p_boot": r["p_boot"],
                             "p_delong_unclustered": dl["p"]})
    pd.DataFrame(cmp_rows).to_csv(C.TABLES / "compare_vs_size.csv", index=False)

    # ---- DCA and calibration curves for key models (S2 pooled and S1 repeat-averaged)
    th = np.round(np.arange(0.05, 0.61, 0.01), 2)
    dca, cal = [], []
    for proto in ("S1", "S2"):
        for m, clf in KEY:
            if (proto, m, clf) not in mats:
                continue
            Pm = mats[(proto, m, clf)]
            y = coh.loc[Pm.index, "y"].values
            p = Pm.values.mean(1)
            nb = S.net_benefit(y, p, th)
            for t, v in zip(th, nb):
                dca.append({"protocol": proto, "model": m, "clf": clf, "threshold": t, "net_benefit": v})
            cc = S.calibration_curve_quantile(y, p, n_bins=5)
            cc["protocol"], cc["model"], cc["clf"] = proto, m, clf
            cal.append(cc)
        y = coh["y"].values
        for t, v in zip(th, S.net_benefit_treat_all(y, th)):
            dca.append({"protocol": proto, "model": "Treat all", "clf": "", "threshold": t, "net_benefit": v})
    pd.DataFrame(dca).to_csv(C.TABLES / "dca.csv", index=False)
    pd.concat(cal).to_csv(C.TABLES / "calibration_curves.csv", index=False)

    audit(coh, mats)


def audit(coh, mats):
    rows = []

    def add(group, name, ref_name, y, p_leaky, groups, p_ref=None, unit="tumour", note=""):
        b = S.cluster_bootstrap_auc(y, p_leaky, groups, NB, C.SEED)
        r = {"group": group, "variant": name, "reference": ref_name, "unit": unit, "n": len(y),
             "auc": b["auc"], "lo": b["lo"], "hi": b["hi"], "note": note}
        if p_ref is not None:
            dd = S.cluster_bootstrap_diff(y, p_leaky, p_ref, groups, NB, C.SEED)
            r.update({"delta": dd["diff"], "delta_lo": dd["lo"], "delta_hi": dd["hi"]})
        rows.append(r)

    # ---- classical (radiomics IT+PT, LR)
    f = C.PREDS / "audit_ml.csv"
    if f.exists():
        A = pd.read_csv(f)
        piv = {v: g.pivot_table(index="case_id", columns="repeat", values="p") for v, g in A.groupby("variant")}
        ref = piv["S1_strict"].reindex(coh.index)
        y, g = coh["y"].values, coh["patient_id"].values
        add("Radiomics", "Strict (patient-grouped CV)", "", y, ref.values, g)
        for v, nm in (("L2_tumour_split", "L2 Tumour-level split (patients ignored)"),
                      ("L3_selection_on_all", "L3 Feature selection on all data")):
            add("Radiomics", nm, "Strict", y, piv[v].reindex(coh.index).values, g, ref.values)
        # L4: different sample (synthetic rows) -> unpaired
        s4 = A[A["variant"] == "L4_SMOTE_before_split"]
        p4 = s4.pivot_table(index="case_id", columns="repeat", values="p")
        meta4 = s4.drop_duplicates("case_id").set_index("case_id").loc[p4.index]
        add("Radiomics", "L4 SMOTE before split", "Strict (no ICC filter)", meta4["y"].values.astype(int), p4.values,
            meta4["patient_id"].values, note="evaluated on real + synthetic tumours; unpaired")
        ref4 = piv["L4_ref_noICC"].reindex(coh.index)
        add("Radiomics", "Strict, no ICC filter (reference for L4/L7-S1)", "", y, ref4.values, g)
        for v, nm in (("L7_S1_H2_all_label_free", "L7 ComBat on all data, label-free (pooled CV)"),
                      ("L7_S1_H3_all_with_label", "L7 ComBat on all data, label as covariate (pooled CV)")):
            add("Radiomics", nm, "Strict, no ICC filter", y, piv[v].reindex(coh.index).values, g, ref4.values)
        lo_ref = piv["L7_LOCO_H0_none"].reindex(coh.index)
        add("Radiomics", "LOCO, no harmonisation", "", y, lo_ref.values, g)
        for v, nm in (("L7_LOCO_H1_train_only", "LOCO, ComBat on training centres only"),
                      ("L7_LOCO_H2_all_label_free", "L7 LOCO, ComBat incl. test centre, label-free"),
                      ("L7_LOCO_H3_all_with_label", "L7 LOCO, ComBat incl. test centre, label as covariate")):
            add("Radiomics", nm, "LOCO, no harmonisation", y, piv[v].reindex(coh.index).values, g, lo_ref.values)
        # L6: Youden threshold re-derived on the test predictions (strict S1 RAD_ITPT-LR)
        d = load_strict("S1")
        d = d[(d["model"] == "RAD_ITPT") & (d["clf"] == "LR")]
        honest, peek = [], []
        for r, gr in d.groupby("repeat"):
            honest.append(S.threshold_metrics(gr["y"].values, (gr["p"] >= gr["thr"]).astype(int).values, 0.5))
            peek.append(S.threshold_metrics(gr["y"].values, gr["p"].values, S.youden_threshold(gr["y"].values, gr["p"].values)))
        for nm, mm in (("Training-derived threshold", honest), ("L6 Threshold tuned on test set", peek)):
            rows.append({"group": "Radiomics", "variant": nm, "reference": "", "unit": "tumour",
                         "sensitivity": np.mean([m["sensitivity"] for m in mm]),
                         "specificity": np.mean([m["specificity"] for m in mm]),
                         "accuracy": np.mean([m["accuracy"] for m in mm]),
                         "balanced_accuracy": np.mean([m["balanced_accuracy"] for m in mm])})
        # L5: best configuration chosen on test AUC (2 repeats)
        G = pd.read_csv(C.PREDS / "audit_ml_L5_grid.csv").join(coh, on="case_id")
        cfg_auc = G.groupby(["config", "repeat"]).apply(lambda x: roc_auc_score(x["y"], x["p"]), include_groups=False)
        cfg_mean = cfg_auc.groupby("config").mean().sort_values(ascending=False)
        cfg_mean.rename("auc_mean_2rep").to_csv(C.TABLES / "audit_L5_configs.csv")
        best = cfg_mean.index[0]
        pb = G[G["config"] == best].pivot_table(index="case_id", columns="repeat", values="p").reindex(coh.index)
        refL5 = ref[[0, 1]]
        add("Radiomics", f"L5 Best of {len(cfg_mean)} configurations (picked on test AUC)", "Strict (repeats 0-1)",
            y, pb.values, g, refL5.values, note=best)
        add("Radiomics", "Strict, repeats 0-1 (reference for L5)", "", y, refL5.values, g)
    # ---- CNN audit
    if (C.PREDS / "cnn_strict.csv").exists():
        st = load_cnn("strict", coh)
        st = st[st["variant"] == "standard"].set_index("case_id").reindex(coh.index)
        y, g = coh["y"].values, coh["patient_id"].values
        add("CNN", "Strict (patient-grouped CV), tumour level", "", y, st["p"].values, g)
        ss = pd.read_csv(C.PREDS / "cnn_strict_slices.csv").join(coh[["patient_id"]], on="case_id")
        add("CNN", "Strict (patient-grouped CV), slice level", "", ss["y"].values, ss["logit"].values,
            ss["patient_id"].values, unit="slice")
        f1 = C.PREDS / "cnn_slice10_slices.csv"
        if f1.exists():
            sl = pd.read_csv(f1).join(coh[["patient_id"]], on="case_id")
            sl = sl.merge(ss[["case_id", "slice_idx", "logit"]].rename(columns={"logit": "logit_strict"}),
                          on=["case_id", "slice_idx"], how="left")          # same slices, strict splits
            assert sl["logit_strict"].notna().all()
            add("CNN", "L1 Slice-level random split (10-fold), slice level", "Strict, slice level", sl["y"].values,
                sl["logit"].values, sl["patient_id"].values, p_ref=sl["logit_strict"].values, unit="slice")
        f1e = C.PREDS / "cnn_slice10_epochs.csv"
        if f1e.exists():          # combined scenario L1 + L5 (fold-mean slice-level test AUC)
            e10 = pd.read_csv(f1e)
            rows.append({"group": "CNN", "unit": "slice", "auc": e10.loc[e10.groupby("fold")["val_auc"].idxmax(),
                                                                          "test_auc"].mean(),
                         "variant": "L1 Slice-level split, epoch chosen on validation (fold-mean test AUC)"})
            rows.append({"group": "CNN", "unit": "slice", "auc": e10.groupby("fold")["test_auc"].max().mean(),
                         "variant": "L1+L5 Slice-level split, epoch chosen on test (fold-mean test AUC)"})
        f2 = C.PREDS / "cnn_tumour5.csv"
        if f2.exists():
            t5 = load_cnn("tumour5", coh).set_index("case_id").reindex(coh.index)
            add("CNN", "L2 Tumour-level split (patients ignored)", "Strict", y, t5["p"].values, g, st["p"].values)
        ep = pd.read_csv(C.PREDS / "cnn_strict_epochs.csv")
        honest = ep.loc[ep.groupby("fold")["val_auc"].idxmax(), "test_auc"].mean()
        peek = ep.groupby("fold")["test_auc"].max().mean()
        rows.append({"group": "CNN", "variant": "Epoch chosen on validation (fold-mean test AUC)", "auc": honest})
        rows.append({"group": "CNN", "variant": "L5 Epoch chosen on test (fold-mean test AUC)", "auc": peek})
        epl = C.PREDS / "cnn_loco_epochs.csv"
        if epl.exists():          # L5 in external validation: epoch picked on the held-out centre
            el = pd.read_csv(epl)
            rows.append({"group": "CNN", "variant": "LOCO, epoch chosen on validation (centre-mean test AUC)",
                         "auc": el.loc[el.groupby("fold")["val_auc"].idxmax(), "test_auc"].mean()})
            rows.append({"group": "CNN", "variant": "L5 LOCO, epoch chosen on held-out centre (centre-mean test AUC)",
                         "auc": el.groupby("fold")["test_auc"].max().mean()})
        if (C.PREDS / "cnn_loco.csv").exists():
            lo = load_cnn("loco", coh)
            a = lo[lo["variant"] == "standard"].set_index("case_id").reindex(coh.index)
            b = lo[lo["variant"] == "AdaBN"].set_index("case_id").reindex(coh.index)
            add("CNN", "LOCO, standard", "", y, a["p"].values, g)
            add("CNN", "L7 LOCO with AdaBN on test-centre images", "LOCO, standard", y, b["p"].values, g, a["p"].values)
    pd.DataFrame(rows).to_csv(C.TABLES / "audit_summary.csv", index=False)
    print(pd.DataFrame(rows)[["group", "variant", "auc", "lo", "hi", "delta", "delta_lo", "delta_hi"]]
          .round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
