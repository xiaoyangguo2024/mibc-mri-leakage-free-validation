"""Leakage-safe modelling: every selection, scaling and tuning step is fit on training data only.

Usage pattern (one outer fold):
    sel = RadiomicsSelector(...).fit(X_tr, y_tr, groups=g_tr, icc=icc_tr)
    clf, params = tune_classifier("LR", sel.transform(X_tr), y_tr, g_tr, seed)
    p_te = clf.predict_proba(sel.transform(X_te))[:, 1]
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy import stats as sps

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------
def strata_of(df, y_col="y", centre_col="centre"):
    """Stratify on centre x label so every fold keeps each centre's case mix."""
    return df[centre_col].astype(str) + "_" + df[y_col].astype(str)


def strict_cv_splits(df, n_splits, n_repeats, seed, y_col="y", group_col="patient_id", centre_col="centre"):
    """Patient-grouped, centre-x-label stratified repeated k-fold. Yields (repeat, fold, train_idx, test_idx)."""
    strata = strata_of(df, y_col, centre_col).values
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(sgkf.split(df, strata, df[group_col].values)):
            yield r, f, tr, te


def tumour_cv_splits(df, n_splits, n_repeats, seed, y_col="y", centre_col="centre"):
    """LEAKY (audit L2): ignores patient grouping -- tumours of one patient can fall on both sides."""
    strata = strata_of(df, y_col, centre_col).values
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(df, strata)):
            yield r, f, tr, te


def loco_splits(df, centres, centre_col="centre"):
    for c in centres:
        te = np.where(df[centre_col].values == c)[0]
        tr = np.where(df[centre_col].values != c)[0]
        yield c, tr, te


def inner_splits(y, groups, n_splits, seed):
    """Grouped inner folds for tuning; falls back to fewer folds if a class is too rare."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    k = n_splits
    while k >= 2:
        try:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            sp = list(sgkf.split(np.zeros(len(y)), y, groups))
            if all(len(np.unique(y[te])) == 2 and len(np.unique(y[tr])) == 2 for tr, te in sp):
                return sp
        except ValueError:
            pass
        k -= 1
    raise ValueError("cannot build inner folds")


# ----------------------------------------------------------------------------------------
# Univariate helpers
# ----------------------------------------------------------------------------------------
def univariate_auc_p(X, y):
    """Vectorised Mann-Whitney: per-column AUC and two-sided normal-approximation p-value."""
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    n1 = y.sum()
    n0 = len(y) - n1
    r = np.apply_along_axis(sps.rankdata, 0, X)
    u = r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2
    aucv = u / (n1 * n0)
    mu = n1 * n0 / 2
    sd = np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
    z = (u - mu) / sd
    p = 2 * sps.norm.sf(np.abs(z))
    return aucv, p


# ----------------------------------------------------------------------------------------
# Radiomics feature selection (fit on training data only)
# ----------------------------------------------------------------------------------------
class RadiomicsSelector:
    """near-zero variance -> ICC filter -> z-score -> Mann-Whitney filter -> |r| pruning -> L1-LR.

    icc: optional pd.Series (feature -> ICC) computed from TRAINING cases only.
    """

    def __init__(self, min_icc=0.75, p_thresh=0.05, min_keep=20, corr_thresh=0.9,
                 max_features=10, seed=0, n_inner=3):
        self.min_icc, self.p_thresh, self.min_keep = min_icc, p_thresh, min_keep
        self.corr_thresh, self.max_features, self.seed, self.n_inner = corr_thresh, max_features, seed, n_inner

    def fit(self, X, y, groups=None, icc=None):
        X = X.astype(float)
        y = np.asarray(y).astype(int)
        cols = X.columns[np.isfinite(X.values).all(axis=0)]
        Xc = X[cols]
        sd = Xc.std(axis=0)
        # near-zero variance or >95% identical values
        top_frac = Xc.apply(lambda s: s.value_counts(normalize=True).iloc[0])
        cols = cols[(sd.values > 1e-8) & (top_frac.values < 0.95)]
        self.n_after_variance_ = len(cols)
        if icc is not None:
            icc = icc.reindex(cols)
            cols = cols[(icc.values >= self.min_icc)]
        self.n_after_icc_ = len(cols)
        self.scaler_ = StandardScaler().fit(X[cols].values)
        Z = self.scaler_.transform(X[cols].values)
        aucv, p = univariate_auc_p(Z, y)
        strength = np.abs(aucv - 0.5)
        keep = np.where(p < self.p_thresh)[0]
        if len(keep) < self.min_keep:
            keep = np.argsort(-strength)[:self.min_keep]
        self.n_after_univariate_ = len(keep)
        # greedy correlation pruning, strongest first
        keep = keep[np.argsort(-strength[keep])]
        C = np.corrcoef(Z[:, keep], rowvar=False) if len(keep) > 1 else np.ones((1, 1))
        chosen = []
        for j in range(len(keep)):
            if all(abs(C[j, i]) < self.corr_thresh for i in chosen):
                chosen.append(j)
        pool = keep[chosen]
        self.n_after_corr_ = len(pool)
        # L1 logistic regression, C chosen by grouped inner CV
        Zp = Z[:, pool]
        g = np.arange(len(y)) if groups is None else np.asarray(groups)
        sp = inner_splits(y, g, self.n_inner, self.seed)
        Cs = np.logspace(-2, 1, 12)
        from sklearn.metrics import roc_auc_score
        means, ses = [], []
        for c in Cs:
            a = []
            for tr, te in sp:
                m = LogisticRegression(penalty="l1", C=c, solver="liblinear", class_weight="balanced",
                                       max_iter=5000).fit(Zp[tr], y[tr])
                d = m.decision_function(Zp[te])
                a.append(roc_auc_score(y[te], d) if np.ptp(d) > 0 else 0.5)
            means.append(np.mean(a))
            ses.append(np.std(a, ddof=1) / np.sqrt(len(a)))
        means, ses = np.array(means), np.array(ses)
        k = int(np.argmax(means))
        # one-standard-error rule (as glmnet's lambda.1se): strongest penalty within 1 SE of the best
        ok = np.where(means >= means[k] - ses[k])[0]
        best_c = Cs[ok.min()]
        # an all-zero model is useless: move to the next weaker penalty that keeps >= 1 feature
        for c in Cs[Cs >= best_c]:
            m = LogisticRegression(penalty="l1", C=c, solver="liblinear", class_weight="balanced",
                                   max_iter=5000).fit(Zp, y)
            if np.any(m.coef_ != 0):
                best_c = c
                break
        m = LogisticRegression(penalty="l1", C=best_c, solver="liblinear", class_weight="balanced",
                               max_iter=5000).fit(Zp, y)
        coef = np.abs(m.coef_.ravel())
        nz = np.where(coef > 0)[0]
        if len(nz) == 0:
            nz = np.argsort(-strength[pool])[:2]
        elif len(nz) > self.max_features:
            nz = nz[np.argsort(-coef[nz])][:self.max_features]
        sel = pool[nz]
        self.lasso_C_ = float(best_c)
        self.cols_ = list(cols)
        self.selected_idx_ = sel
        self.selected_ = [cols[i] for i in sel]
        return self

    def transform(self, X):
        Z = self.scaler_.transform(X[self.cols_].astype(float).values)
        return pd.DataFrame(Z[:, self.selected_idx_], index=X.index, columns=self.selected_)


# ----------------------------------------------------------------------------------------
# Classifier zoo (the eight classifiers of the reference study)
# ----------------------------------------------------------------------------------------
def classifier_zoo(seed, pos_weight=1.0):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    return {
        "LR": (LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs"),
               {"C": [0.01, 0.1, 1.0, 10.0]}),
        "SVM": (SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
                {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}),
        "KNN": (KNeighborsClassifier(weights="distance"), {"n_neighbors": [5, 9, 15, 25]}),
        "RF": (RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1),
               {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]}),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1),
                       {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]}),
        "XGBoost": (XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                  scale_pos_weight=pos_weight, random_state=seed, n_jobs=1, verbosity=0),
                    {"max_depth": [2, 3], "learning_rate": [0.03, 0.1]}),
        "LightGBM": (LGBMClassifier(n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                    class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1),
                     {"num_leaves": [4, 8], "learning_rate": [0.03, 0.1], "min_child_samples": [10, 20]}),
        "MLP": (MLPClassifier(max_iter=3000, random_state=seed, early_stopping=False),
                {"hidden_layer_sizes": [(16,), (32, 16)], "alpha": [1e-3, 1e-1]}),
    }


CLASSIFIERS = ["LR", "SVM", "KNN", "RF", "ExtraTrees", "XGBoost", "LightGBM", "MLP"]


def tune_classifier(name, X, y, groups, seed, n_inner=3, n_jobs=1):
    y = np.asarray(y).astype(int)
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())
    est, grid = classifier_zoo(seed, pos_weight)[name]
    sp = inner_splits(y, groups, n_inner, seed)
    gs = GridSearchCV(clone(est), grid, cv=sp, scoring="roc_auc", n_jobs=n_jobs, refit=True, error_score="raise")
    gs.fit(np.asarray(X, float), y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def all_grid_fits(name, X_tr, y_tr, X_te, seed):
    """AUDIT L5 helper: fit EVERY grid point on the training data and return test probabilities
    for each, so that 'pick the configuration with the best test AUC' can be simulated."""
    y_tr = np.asarray(y_tr).astype(int)
    pos_weight = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
    est, grid = classifier_zoo(seed, pos_weight)[name]
    from sklearn.model_selection import ParameterGrid
    out = []
    for prm in ParameterGrid(grid):
        m = clone(est).set_params(**prm).fit(np.asarray(X_tr, float), y_tr)
        out.append((prm, m.predict_proba(np.asarray(X_te, float))[:, 1]))
    return out


# ----------------------------------------------------------------------------------------
# Linear probe for high-dimensional deep features
# ----------------------------------------------------------------------------------------
def tune_linear_probe(X, y, groups, seed, n_inner=3, n_jobs=1):
    """z-score -> PCA -> L2-LR, with n_components and C tuned on grouped inner folds."""
    y = np.asarray(y).astype(int)
    pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(random_state=seed)),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=5000))])
    nmax = min(np.asarray(X).shape[1], int(len(y) * 0.6) - 1)
    grid = {"pca__n_components": [k for k in (5, 10, 20, 40) if k <= nmax], "lr__C": [0.001, 0.01, 0.1, 1.0]}
    sp = inner_splits(y, groups, n_inner, seed)
    gs = GridSearchCV(pipe, grid, cv=sp, scoring="roc_auc", n_jobs=n_jobs, refit=True)
    gs.fit(np.asarray(X, float), y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)
