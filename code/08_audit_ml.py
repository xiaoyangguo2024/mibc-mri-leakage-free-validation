"""08 - Leakage audit for the classical pipelines (radiomics IT+PT, and frozen-feature linear probes).

Each variant changes ONE step of the strict protocol (06) and keeps everything else identical.
Reference = strict S1 (patient-grouped 5x5 CV) or strict S2 (LOCO).

  L2  tumour-level splits that ignore patient grouping (outer and inner)
  L3  feature selection (univariate + correlation + LASSO) on all 275 tumours before CV
  L4  SMOTE oversampling of the whole dataset before splitting
  L5  'best of many': every classifier x hyper-parameter x feature-number configuration is
      evaluated on the outer test folds and the best one is reported
  L7  ComBat harmonisation:  H1 fit on training centres only (strict)
                             H2 fit on all centres incl. the test centre, label-free (transductive)
                             H3 fit on all centres incl. test, outcome as protected covariate (label leakage)
Outputs: results/predictions/audit_ml.csv (long), results/tables/audit_ml_L5_grid.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bcmri import config as C
from bcmri import data as D
from bcmri import ml

N_JOBS = int(__import__("os").environ.get("BCMRI_JOBS", "4"))


def radiomics_fold(R, coh, tr, te, groups, seed, icc, selector_fixed=None, clf="LR", max_features=None):
    """Selection in-fold (unless selector_fixed is given) + tuned classifier; returns test probabilities."""
    ids_tr, ids_te = R.index[tr], R.index[te]
    y_tr = coh["y"].values[tr]
    g_tr = groups[tr]
    cols = list(R.columns)
    if selector_fixed is None:
        sel = ml.RadiomicsSelector(max_features=max_features or C.MAX_FEATURES, seed=seed).fit(
            R.loc[ids_tr, cols], y_tr, groups=g_tr, icc=icc(ids_tr, cols) if icc else None)
    else:
        sel = selector_fixed
    Xtr, Xte = sel.transform(R.loc[ids_tr, cols]), sel.transform(R.loc[ids_te, cols])
    est, prm, sc = ml.tune_classifier(clf, Xtr, y_tr, g_tr, seed, C.INNER_FOLDS, N_JOBS)
    return est.predict_proba(Xte.values)[:, 1], sel


def run_cv(name, splits, R, coh, groups, icc, out, **kw):
    t0 = time.time()
    for r, f, tr, te in splits:
        p, _ = radiomics_fold(R, coh, tr, te, groups, C.SEED + 100 * r + f, icc, **kw)
        for cid, pv in zip(R.index[te], p):
            out.append({"variant": name, "repeat": r, "fold": f, "case_id": cid, "p": float(pv)})
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)


def combat(X, batch, label=None):
    from neuroCombat import neuroCombat
    keep = X.columns[(X.groupby(batch.values).transform("std") > 1e-8).all(axis=0).values]
    covars = pd.DataFrame({"batch": batch.values})
    cat = []
    if label is not None:
        covars["label"] = label.values
        cat = ["label"]
    res = neuroCombat(dat=X[keep].T.values, covars=covars, batch_col="batch", categorical_cols=cat or None)
    out = X.copy()
    out[keep] = res["data"].T
    return out


def main():
    coh = D.load_cohort()
    R_all = D.load_radiomics("orig").reindex(coh.index)
    cols = [c for c in R_all.columns if c.startswith(("IT_", "PT_"))]
    R = R_all[cols].replace([np.inf, -np.inf], np.nan)
    R = R.loc[:, R.notna().all(axis=0)]
    icc = D.ICCProvider()
    y = coh["y"].values
    pid = coh["patient_id"].values
    cases = coh.reset_index()
    out = []
    strict_splits = list(ml.strict_cv_splits(cases, C.OUTER_FOLDS, C.OUTER_REPEATS, C.SEED))

    # reference strict S1 (re-run here with LR so that every variant shares code and seeds)
    run_cv("S1_strict", strict_splits, R, coh, pid, icc, out)
    # L2: ignore patient grouping (outer + inner)
    tum_splits = list(ml.tumour_cv_splits(cases, C.OUTER_FOLDS, C.OUTER_REPEATS, C.SEED))
    run_cv("L2_tumour_split", tum_splits, R, coh, coh.index.values, icc, out)
    # L3: selection on all data before CV
    sel_all = ml.RadiomicsSelector(max_features=C.MAX_FEATURES, seed=C.SEED).fit(R, y, groups=pid, icc=icc(R.index, list(R.columns)))
    print("L3 selected on all data:", sel_all.selected_, flush=True)
    run_cv("L3_selection_on_all", strict_splits, R, coh, pid, icc, out, selector_fixed=sel_all)

    # L4: SMOTE before splitting (synthetic MIBC tumours get their own 'patient' ids)
    from imblearn.over_sampling import SMOTE
    Xs, ys = SMOTE(random_state=C.SEED, k_neighbors=5).fit_resample(R.values, y)
    n_syn = len(ys) - len(y)
    ids = list(R.index) + [f"SYN{i:03d}" for i in range(n_syn)]
    Rs = pd.DataFrame(Xs, columns=R.columns, index=ids)
    cohs = pd.DataFrame({"y": ys, "patient_id": list(pid) + ids[len(y):],
                         "centre": list(coh["centre"]) + ["SYN"] * n_syn}, index=ids)
    cohs.index.name = "case_id"
    syn_splits = list(ml.strict_cv_splits(cohs.reset_index(), C.OUTER_FOLDS, C.OUTER_REPEATS, C.SEED))
    run_cv("L4_SMOTE_before_split", syn_splits, Rs, cohs, cohs["patient_id"].values, None, out)
    # (ICC filter skipped for synthetic rows -- no perturbed masks exist for them; a strict SMOTE-in-fold
    #  reference without ICC is also run so the comparison isolates the oversampling step)
    run_cv("L4_ref_noICC", strict_splits, R, coh, pid, None, out)

    # L5: best of many configurations, chosen by test AUC
    from sklearn.model_selection import ParameterGrid
    grid_rows = []
    t0 = time.time()
    for r, f, tr, te in strict_splits[:C.OUTER_FOLDS * 2]:          # two repeats keep the run time moderate
        ids_tr, ids_te = R.index[tr], R.index[te]
        for k in (5, 10, 20):
            sel = ml.RadiomicsSelector(max_features=k, seed=C.SEED).fit(R.loc[ids_tr], y[tr], groups=pid[tr],
                                                                        icc=icc(ids_tr, list(R.columns)))
            Xtr, Xte = sel.transform(R.loc[ids_tr]), sel.transform(R.loc[ids_te])
            for clf in ml.CLASSIFIERS:
                for prm, p in ml.all_grid_fits(clf, Xtr.values, y[tr], Xte.values, C.SEED):
                    cfg = f"{clf}|k={k}|{sorted(prm.items())}"
                    for cid, pv in zip(ids_te, p):
                        grid_rows.append({"repeat": r, "fold": f, "config": cfg, "case_id": cid, "p": float(pv)})
        print(f"L5 r{r} f{f}: {time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(grid_rows)
    G.to_csv(C.PREDS / "audit_ml_L5_grid.csv", index=False)

    # L7: ComBat variants under LOCO (and S1 for H2/H3)
    batch = coh["centre"]
    label = coh["y"]
    for c, tr, te in ml.loco_splits(coh, C.CENTRES):
        variants = {
            "H0_none": R,
            "H1_train_only": pd.concat([combat(R.iloc[tr], batch.iloc[tr]), R.iloc[te]]).reindex(R.index),
            "H2_all_label_free": combat(R, batch),
            "H3_all_with_label": combat(R, batch, label),
        }
        for name, Rv in variants.items():
            p, _ = radiomics_fold(Rv, coh, tr, te, pid, C.SEED, None)
            for cid, pv in zip(R.index[te], p):
                out.append({"variant": f"L7_LOCO_{name}", "repeat": 0, "fold": c, "case_id": cid, "p": float(pv)})
        print(f"L7 LOCO {c} done", flush=True)
    for name, Rv in {"H2_all_label_free": combat(R, batch), "H3_all_with_label": combat(R, batch, label)}.items():
        run_cv(f"L7_S1_{name}", strict_splits, Rv, coh, pid, None, out)
    run_cv("L7_S1_H0_none_noICC", strict_splits, R, coh, pid, None, out)

    O = pd.DataFrame(out).merge(coh[["y", "centre", "patient_id"]], left_on="case_id", right_index=True, how="left")
    syn = O["case_id"].str.startswith("SYN")
    O.loc[syn, "y"] = 1
    O.loc[syn, "patient_id"] = O.loc[syn, "case_id"]
    O.loc[syn, "centre"] = "SYN"
    O.to_csv(C.PREDS / "audit_ml.csv", index=False)
    for v, g in O.groupby("variant"):
        aucs = [roc_auc_score(gg["y"], gg["p"]) for _, gg in g.groupby("repeat") if gg["y"].nunique() == 2]
        print(f"{v:32s} mean AUC over repeats {np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
