"""Statistics used throughout the paper.

All confidence intervals resample whole patients (clusters), because some patients
contribute several tumours. DeLong is provided for reference only: it assumes
independent tumours.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------------------------------------------------------------------
# Basic metrics
# ----------------------------------------------------------------------------------------
def auc(y, p):
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float)))


def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), np.asarray(p, float))
    j = tpr - fpr
    k = int(np.argmax(j))
    t = thr[k]
    if not np.isfinite(t):           # sklearn puts +inf first
        t = np.nextafter(np.max(p), np.inf)
    return float(t)


def threshold_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p, float) >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    div = lambda a, b: float(a / b) if b > 0 else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    return {"threshold": float(thr), "sensitivity": sens, "specificity": spec,
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "accuracy": div(tp + tn, len(y)), "balanced_accuracy": (sens + spec) / 2,
            "f1": div(2 * tp, 2 * tp + fp + fn), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def calibration_intercept_slope(y, p, eps=1e-6):
    """Calibration-in-the-large (intercept with the linear predictor as offset) and slope."""
    import statsmodels.api as sm
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    lp = logit(p)
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:
        slope = np.nan
    try:
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        citl = np.nan
    return float(citl), float(slope)


def calibration_curve_quantile(y, p, n_bins=5):
    """Observed vs mean predicted risk in quantile bins (for plotting)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    g = df.groupby("bin")
    out = pd.DataFrame({"p_mean": g["p"].mean(), "obs": g["y"].mean(), "n": g.size()})
    out["se"] = np.sqrt(out["obs"] * (1 - out["obs"]) / out["n"])
    return out.reset_index(drop=True)


def net_benefit(y, p, thresholds):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    n = len(y)
    nb = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb.append(tp / n - fp / n * t / (1 - t))
    return np.asarray(nb)


def net_benefit_treat_all(y, thresholds):
    prev = float(np.mean(np.asarray(y).astype(int)))
    t = np.asarray(thresholds, float)
    return prev - (1 - prev) * t / (1 - t)


# ----------------------------------------------------------------------------------------
# Patient-level (cluster) bootstrap
# ----------------------------------------------------------------------------------------
def _cluster_index(groups):
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    members = [order[s:s + c] for s, c in zip(starts, counts)]
    return members


def bootstrap_indices(groups, n_boot, seed):
    """Yield index arrays, each a resample of whole clusters with replacement."""
    members = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    k = len(members)
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        yield np.concatenate([members[i] for i in pick])


def cluster_bootstrap_auc(y, p, groups, n_boot=2000, seed=0):
    """AUC with patient-level percentile CI. `p` may be 1-D or (n, R) for R repeats
    (then the statistic is the mean AUC over repeats, evaluated on the same resample)."""
    y = np.asarray(y).astype(int)
    P = np.asarray(p, float)
    if P.ndim == 1:
        P = P[:, None]
    est = float(np.mean([auc(y, P[:, r]) for r in range(P.shape[1])]))
    vals = []
    for idx in bootstrap_indices(groups, n_boot, seed):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(np.mean([roc_auc_score(yb, P[idx, r]) for r in range(P.shape[1])]))
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auc": est, "lo": float(lo), "hi": float(hi),
            "se_logit": float(np.std(logit(np.clip(vals, 1e-4, 1 - 1e-4)), ddof=1)), "n_boot": len(vals)}


def cluster_bootstrap_diff(y, p1, p2, groups, n_boot=2000, seed=0):
    """Paired difference AUC(p1) - AUC(p2), patient-level bootstrap CI and two-sided p."""
    y = np.asarray(y).astype(int)
    P1 = np.asarray(p1, float)
    P2 = np.asarray(p2, float)
    P1 = P1[:, None] if P1.ndim == 1 else P1
    P2 = P2[:, None] if P2.ndim == 1 else P2
    f = lambda yy, A: np.mean([roc_auc_score(yy, A[:, r]) for r in range(A.shape[1])])
    est = float(f(y, P1) - f(y, P2))
    d = []
    for idx in bootstrap_indices(groups, n_boot, seed):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        d.append(f(yb, P1[idx]) - f(yb, P2[idx]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": est, "lo": float(lo), "hi": float(hi), "p_boot": float(min(1.0, p))}


def cluster_bootstrap_metric(fn, y, p, groups, n_boot=2000, seed=0):
    """Generic percentile CI for any fn(y, p) -> float."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    est = fn(y, p)
    vals = []
    for idx in bootstrap_indices(groups, n_boot, seed):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        try:
            vals.append(fn(yb, p[idx]))
        except Exception:
            pass
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


# ----------------------------------------------------------------------------------------
# DeLong (Sun & Xu 2014 fast algorithm) -- reference only, assumes independent tumours
# ----------------------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds, m):
    n = preds.shape[1] - m
    k = preds.shape[0]
    tx = np.array([_midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([_midrank(preds[r, :]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")       # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z),
            "p": float(2 * sps.norm.sf(abs(z)))}


def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    aucs, cov = _fast_delong(np.asarray(p, float)[order][None, :], m)
    se = float(np.sqrt(cov[0, 0]))
    zq = sps.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(aucs[0] - zq * se), float(aucs[0] + zq * se), se


# ----------------------------------------------------------------------------------------
# Random-effects pooling of centre-specific AUCs (logit scale)
# ----------------------------------------------------------------------------------------
def random_effects_auc(aucs, se_logit):
    """DerSimonian-Laird tau^2, Hartung-Knapp-Sidik-Jonkman CI (recommended with few
    centres), Higgins prediction interval. Inputs: centre AUCs and SEs on the logit scale."""
    a = np.clip(np.asarray(aucs, float), 1e-3, 1 - 1e-3)
    th = logit(a)
    v = np.asarray(se_logit, float) ** 2
    k = len(th)
    if not np.all(np.isfinite(v)) or np.any(v <= 0):        # degenerate centre (e.g. constant predictions)
        return {key: np.nan for key in ("auc", "lo_dl", "hi_dl", "lo", "hi", "tau2", "I2", "Q", "p_Q",
                                        "pi_lo", "pi_hi")} | {"k": k}
    w = 1 / v
    th_fe = np.sum(w * th) / np.sum(w)
    q = float(np.sum(w * (th - th_fe) ** 2))
    df = k - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (v + tau2)
    th_re = np.sum(wr * th) / np.sum(wr)
    se_dl = np.sqrt(1 / np.sum(wr))
    # HKSJ
    se_hk = np.sqrt(np.sum(wr * (th - th_re) ** 2) / (df * np.sum(wr))) if df > 0 else se_dl
    se_hk = max(se_hk, se_dl)       # conservative variant (never narrower than DL)
    tq = sps.t.ppf(0.975, df) if df > 0 else np.nan
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    out = {"k": k, "auc": float(expit(th_re)),
           "lo_dl": float(expit(th_re - 1.96 * se_dl)), "hi_dl": float(expit(th_re + 1.96 * se_dl)),
           "lo": float(expit(th_re - tq * se_hk)), "hi": float(expit(th_re + tq * se_hk)),
           "tau2": float(tau2), "I2": float(i2), "Q": q, "p_Q": float(sps.chi2.sf(q, df)) if df > 0 else np.nan}
    if k > 2:
        tp = sps.t.ppf(0.975, k - 2)
        half = tp * np.sqrt(tau2 + se_dl ** 2)
        out["pi_lo"], out["pi_hi"] = float(expit(th_re - half)), float(expit(th_re + half))
    else:
        out["pi_lo"] = out["pi_hi"] = np.nan
    return out


# ----------------------------------------------------------------------------------------
# Intraclass correlation for feature robustness
# ----------------------------------------------------------------------------------------
def icc_3_1(mat):
    """ICC(3,1), two-way mixed, consistency, single measurement. mat: (n_subjects, k_raters)."""
    x = np.asarray(mat, float)
    n, k = x.shape
    gm = x.mean()
    ss_rows = k * np.sum((x.mean(axis=1) - gm) ** 2)
    ss_cols = n * np.sum((x.mean(axis=0) - gm) ** 2)
    ss_tot = np.sum((x - gm) ** 2)
    ss_err = ss_tot - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    return float((ms_rows - ms_err) / denom) if denom > 0 else np.nan


def icc_per_feature(stack):
    """stack: array (n_subjects, k_versions, n_features) -> ICC(3,1) per feature."""
    s = np.asarray(stack, float)
    n, k, f = s.shape
    gm = s.mean(axis=(0, 1))
    ss_rows = k * np.sum((s.mean(axis=1) - gm) ** 2, axis=0)
    ss_cols = n * np.sum((s.mean(axis=0) - gm) ** 2, axis=0)
    ss_tot = np.sum((s - gm) ** 2, axis=(0, 1))
    ss_err = ss_tot - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    with np.errstate(invalid="ignore", divide="ignore"):
        icc = (ms_rows - ms_err) / denom
    icc[~np.isfinite(icc)] = np.nan
    return icc
