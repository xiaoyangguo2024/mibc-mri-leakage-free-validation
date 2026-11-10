"""06 - Strict (leakage-free) evaluation of the classical models.

Protocols
  S1  patient-grouped, centre x label stratified 5-fold CV, 5 repeats (internal)
  S2  leave-one-centre-out (external, primary endpoint)
  S3  develop on C1, test on C2, C3, C4 (single-centre development design)
Models
  RAD_IT, RAD_ITPT  radiomics (intratumoural; + 3-mm ring) x 8 classifiers
  DL_<backbone>     frozen deep features, linear probe (z-score -> PCA -> L2-LR)
  DLR               fusion: ICC-filtered radiomics + PCs of the inner-CV-best backbone x 8 classifiers
  LATE              late fusion: mean of RAD_ITPT-LR and the best DL probe probabilities
  SIZE              logistic regression on log volume + maximum 2-D diameter
  CENTRE            (S1 only) training prevalence of the case's centre -- confounding check
Every selection / scaling / tuning step is fit on the training part of each split only.

Outputs: results/predictions/strict_{S1,S2,S3}.csv (long format), results/tables/selected_features_strict.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from bcmri import config as C
from bcmri import data as D
from bcmri import ml
from bcmri.stats import youden_threshold

N_JOBS = int(__import__("os").environ.get("BCMRI_JOBS", "4"))


def train_oof_threshold(est, X, y, groups, seed):
    """Youden threshold from grouped inner-CV predictions on the TRAINING data."""
    sp = ml.inner_splits(y, groups, C.INNER_FOLDS, seed)
    p = cross_val_predict(est, np.asarray(X, float), y, cv=sp, method="predict_proba")[:, 1]
    return youden_threshold(y, p)


def run_split(tag, tr, te, coh, R, deep, icc, seed, records, sel_log):
    ids_tr, ids_te = coh.index[tr], coh.index[te]
    y_tr = coh["y"].values[tr]
    g_tr = coh["patient_id"].values[tr]

    def put(model, clf, p, thr, extra=None):
        for cid, pv in zip(ids_te, p):
            rec = {**tag, "case_id": cid, "model": model, "clf": clf, "p": float(pv), "thr": thr}
            if extra:
                rec.update(extra)
            records.append(rec)

    # ---- radiomics (two regions) x 8 classifiers
    for fs, prefix in (("RAD_IT", ("IT_",)), ("RAD_ITPT", ("IT_", "PT_"))):
        cols = [c for c in R.columns if c.startswith(prefix)]
        sel = ml.RadiomicsSelector(max_features=C.MAX_FEATURES, seed=seed).fit(
            R.loc[ids_tr, cols], y_tr, groups=g_tr, icc=icc(ids_tr, cols))
        Xtr, Xte = sel.transform(R.loc[ids_tr, cols]), sel.transform(R.loc[ids_te, cols])
        sel_log.append({**tag, "model": fs, "n_var": sel.n_after_variance_, "n_icc": sel.n_after_icc_,
                        "n_uni": sel.n_after_univariate_, "n_corr": sel.n_after_corr_, "lasso_C": sel.lasso_C_,
                        "selected": "|".join(sel.selected_)})
        for clf in ml.CLASSIFIERS:
            est, prm, sc = ml.tune_classifier(clf, Xtr, y_tr, g_tr, seed, C.INNER_FOLDS, N_JOBS)
            thr = train_oof_threshold(est, Xtr, y_tr, g_tr, seed)
            put(fs, clf, est.predict_proba(Xte.values)[:, 1], thr, {"inner_auc": sc, "params": str(prm)})

    # ---- frozen deep features, linear probe
    best_b, best_sc, dl_p = None, -1, {}
    for b, Xd in deep.items():
        est, prm, sc = ml.tune_linear_probe(Xd.loc[ids_tr].values, y_tr, g_tr, seed, C.INNER_FOLDS, N_JOBS)
        thr = train_oof_threshold(est, Xd.loc[ids_tr].values, y_tr, g_tr, seed)
        p = est.predict_proba(Xd.loc[ids_te].values)[:, 1]
        dl_p[b] = p
        put(f"DL_{b}", "LR", p, thr, {"inner_auc": sc, "params": str(prm)})
        if sc > best_sc:
            best_b, best_sc, best_ncomp = b, sc, prm["pca__n_components"]

    # ---- early fusion (DLR): ICC-filtered radiomics + PCs of the best backbone
    cols = [c for c in R.columns if c.startswith(("IT_", "PT_"))]
    Xd = deep[best_b]
    sc_d = StandardScaler().fit(Xd.loc[ids_tr].values)
    pca = PCA(n_components=min(best_ncomp, 20), random_state=seed).fit(sc_d.transform(Xd.loc[ids_tr].values))
    pc_names = [f"PC{i + 1}_{best_b}" for i in range(pca.n_components_)]
    pcs_tr = pd.DataFrame(pca.transform(sc_d.transform(Xd.loc[ids_tr].values)), index=ids_tr, columns=pc_names)
    pcs_te = pd.DataFrame(pca.transform(sc_d.transform(Xd.loc[ids_te].values)), index=ids_te, columns=pc_names)
    pool_tr = R.loc[ids_tr, cols].join(pcs_tr)
    pool_te = R.loc[ids_te, cols].join(pcs_te)
    icc_s = pd.concat([icc(ids_tr, cols), pd.Series(1.0, index=pc_names)])
    sel = ml.RadiomicsSelector(max_features=C.MAX_FEATURES, seed=seed).fit(pool_tr, y_tr, groups=g_tr, icc=icc_s)
    Xtr, Xte = sel.transform(pool_tr), sel.transform(pool_te)
    sel_log.append({**tag, "model": "DLR", "backbone": best_b, "n_var": sel.n_after_variance_,
                    "n_icc": sel.n_after_icc_, "n_uni": sel.n_after_univariate_, "n_corr": sel.n_after_corr_,
                    "lasso_C": sel.lasso_C_, "selected": "|".join(sel.selected_)})
    for clf in ml.CLASSIFIERS:
        est, prm, sc = ml.tune_classifier(clf, Xtr, y_tr, g_tr, seed, C.INNER_FOLDS, N_JOBS)
        thr = train_oof_threshold(est, Xtr, y_tr, g_tr, seed)
        put("DLR", clf, est.predict_proba(Xte.values)[:, 1], thr,
            {"inner_auc": sc, "params": str(prm), "backbone": best_b})

    # ---- late fusion: RAD_ITPT-LR + best DL probe (simple mean of probabilities)
    rad_lr = [r for r in records if r.get("model") == "RAD_ITPT" and r.get("clf") == "LR"
              and all(r.get(k) == v for k, v in tag.items())]
    p_rad = pd.Series({r["case_id"]: r["p"] for r in rad_lr}).reindex(ids_te).values
    put("LATE", "mean", (p_rad + dl_p[best_b]) / 2, np.nan, {"backbone": best_b})

    # ---- size baseline
    S = D.size_features(R)
    sc_s = StandardScaler().fit(S.loc[ids_tr])
    est, prm, sc = ml.tune_classifier("LR", sc_s.transform(S.loc[ids_tr]), y_tr, g_tr, seed, C.INNER_FOLDS, 1)
    thr = train_oof_threshold(est, sc_s.transform(S.loc[ids_tr]), y_tr, g_tr, seed)
    put("SIZE", "LR", est.predict_proba(sc_s.transform(S.loc[ids_te]))[:, 1], thr, {"inner_auc": sc})

    # ---- centre-prevalence baseline (meaningful only when the test centre is seen in training)
    prev = coh.iloc[tr].groupby("centre")["y"].mean()
    p_c = coh.iloc[te]["centre"].map(prev).fillna(coh["y"].values[tr].mean()).values
    put("CENTRE", "prevalence", p_c, np.nan)


def main():
    protocols = sys.argv[1:] or ["S1", "S2", "S3"]
    coh = D.load_cohort()
    R = D.load_radiomics("orig").reindex(coh.index)
    deep = {b: D.load_deep(b).reindex(coh.index) for b in D.BACKBONES}
    icc = D.ICCProvider()
    seed = C.SEED
    for proto in protocols:
        t0 = time.time()
        records, sel_log = [], []
        if proto == "S1":
            splits = [({"protocol": "S1", "repeat": r, "fold": f}, tr, te)
                      for r, f, tr, te in ml.strict_cv_splits(coh.reset_index(), C.OUTER_FOLDS, C.OUTER_REPEATS, seed)]
        elif proto == "S2":
            splits = [({"protocol": "S2", "repeat": 0, "fold": c}, tr, te) for c, tr, te in ml.loco_splits(coh, C.CENTRES)]
        elif proto == "S3":
            tr = np.where(coh["centre"].values == "C1")[0]
            te = np.where(coh["centre"].values != "C1")[0]
            splits = [({"protocol": "S3", "repeat": 0, "fold": "C1->C2-4"}, tr, te)]
        for k, (tag, tr, te) in enumerate(splits):
            # patient grouping sanity check
            assert not set(coh["patient_id"].values[tr]) & set(coh["patient_id"].values[te])
            run_split(tag, tr, te, coh, R, deep, icc, seed + k, records, sel_log)
            print(f"[{proto}] split {k + 1}/{len(splits)} done ({time.time() - t0:.0f}s)", flush=True)
        out = pd.DataFrame(records)
        out = out.merge(coh[["y", "centre", "patient_id"]], left_on="case_id", right_index=True, how="left")
        out.to_csv(C.PREDS / f"strict_{proto}.csv", index=False)
        pd.DataFrame(sel_log).to_csv(C.TABLES / f"selected_features_strict_{proto}.csv", index=False)
        print(f"[{proto}] finished in {time.time() - t0:.0f}s; {len(out)} prediction rows", flush=True)


if __name__ == "__main__":
    main()
