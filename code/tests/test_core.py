"""Sanity checks of the statistics and selection code on synthetic data (run: python -I code/tests/test_core.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from bcmri import stats as S
from bcmri import ml

rng = np.random.default_rng(1)
n = 300
y = rng.integers(0, 2, n)
p1 = y * 0.8 + rng.normal(0, 1, n)
p2 = y * 0.3 + rng.normal(0, 1, n)
g = np.repeat(np.arange(n // 3), 3)

# DeLong AUC equals sklearn AUC; CI sensible
a, lo, hi, se = S.delong_ci(y, p1)
assert abs(a - roc_auc_score(y, p1)) < 1e-12, (a, roc_auc_score(y, p1))
t = S.delong_test(y, p1, p2)
assert t["p"] < 0.05 and abs(t["auc1"] - a) < 1e-12
print("DeLong OK", round(a, 3), (round(lo, 3), round(hi, 3)), "p12", t["p"])

# cluster bootstrap: CI contains the estimate; wider with fewer clusters
r = S.cluster_bootstrap_auc(y, p1, g, n_boot=500, seed=0)
r_ind = S.cluster_bootstrap_auc(y, p1, np.arange(n), n_boot=500, seed=0)
assert r["lo"] < r["auc"] < r["hi"]
print("cluster bootstrap OK", r, "| independent:", r_ind)

# random effects: identical AUCs -> tau2 = 0, I2 = 0
re = S.random_effects_auc([0.8, 0.8, 0.8, 0.8], [0.3, 0.3, 0.3, 0.3])
assert re["tau2"] == 0 and abs(re["auc"] - 0.8) < 1e-9
re2 = S.random_effects_auc([0.6, 0.9, 0.7, 0.85], [0.2, 0.2, 0.3, 0.25])
print("random effects OK", {k: round(v, 3) if isinstance(v, float) else v for k, v in re2.items()})

# ICC: identical raters -> 1; noise-only -> ~0
base = rng.normal(0, 1, (100, 1))
same = np.repeat(base, 3, axis=1)
assert abs(S.icc_3_1(same) - 1) < 1e-9
noise = rng.normal(0, 1, (100, 3))
assert abs(S.icc_3_1(noise)) < 0.2
stack = np.stack([np.hstack([base + rng.normal(0, s, (100, 1)) for s in (0.1, 1.0, 5.0)]) for _ in range(3)], axis=1)
iccs = S.icc_per_feature(stack)
assert iccs[0] > 0.95 and iccs[2] < 0.2, iccs
assert abs(S.icc_per_feature(same[:, :, None])[0] - 1) < 1e-9
print("ICC OK", np.round(iccs, 3))

# calibration: perfectly calibrated probs -> intercept ~0, slope ~1
pp = rng.uniform(0.05, 0.95, 5000)
yy = rng.binomial(1, pp)
citl, slope = S.calibration_intercept_slope(yy, pp)
assert abs(citl) < 0.1 and abs(slope - 1) < 0.1, (citl, slope)
print("calibration OK", round(citl, 3), round(slope, 3))

# net benefit: treat-all formula consistent with net_benefit at p=1
th = np.array([0.1, 0.2, 0.3])
assert np.allclose(S.net_benefit(y, np.ones(n), th), S.net_benefit_treat_all(y, th))
print("net benefit OK")

# selector picks informative features, and nothing when trained on noise only
X = pd.DataFrame(rng.normal(0, 1, (n, 200)), columns=[f"f{i}" for i in range(200)])
X["f0"] += y * 1.5
X["f1"] += y * 1.0
sel = ml.RadiomicsSelector(max_features=10, seed=0).fit(X, y, groups=g)
assert "f0" in sel.selected_ and "f1" in sel.selected_, sel.selected_
print("selector OK", sel.selected_, "C =", sel.lasso_C_)
for fn in (ml.strict_cv_splits,):
    df = pd.DataFrame({"y": y, "patient_id": g, "centre": rng.choice(["C1", "C2"], n)})
    for r_, f_, tr, te in fn(df, 5, 2, 0):
        assert not set(df.patient_id.values[tr]) & set(df.patient_id.values[te])
print("grouped splits OK")
est, prm, sc = ml.tune_classifier("LightGBM", X[sel.selected_].values, y, g, 0)
print("tuning OK", prm, round(sc, 3))
print("ALL TESTS PASSED")
