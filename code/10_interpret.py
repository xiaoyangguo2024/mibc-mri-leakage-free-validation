"""10 - Interpretability.

  A  Radiomics: selection frequency of each feature across the 25 strict S1 folds; final pre-specified
     model (RAD_ITPT selection + L2-logistic regression) refit on all tumours for SHAP (exact linear SHAP)
  B  CNN (strict S1 fold models, each applied only to its own test fold):
     - Grad-CAM on layer3 (8x8 for 128-px input): fraction of attention inside the tumour, the 3-mm ring,
       and elsewhere, against the area fractions of those regions (enrichment = energy share / area share)
     - region ablation: tumour-level AUC when the tumour, the ring, or everything outside the tumour is
       replaced by a heavily blurred copy of the image (texture removed, mean intensity kept)
Outputs: results/tables/interp_*.csv, results/models/final_rad_itpt_lr.pkl, data/processed/gradcam_examples.npz
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bcmri import config as C
from bcmri import data as D
from bcmri import ml

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def part_a():
    coh = D.load_cohort()
    R = D.load_radiomics("orig").reindex(coh.index)
    cols = [c for c in R.columns if c.startswith(("IT_", "PT_"))]
    R = R[cols]
    R = R.loc[:, np.isfinite(R.values).all(axis=0)]
    sel_log = pd.read_csv(C.TABLES / "selected_features_strict_S1.csv")
    s = sel_log[sel_log["model"] == "RAD_ITPT"]["selected"].str.split("|").explode()
    freq = s.value_counts().rename("n_folds").to_frame()
    freq["fraction"] = freq["n_folds"] / (sel_log["model"] == "RAD_ITPT").sum()
    freq.index.name = "feature"
    freq.to_csv(C.TABLES / "interp_selection_frequency.csv")
    icc = D.ICCProvider()
    y = coh["y"].values
    sel = ml.RadiomicsSelector(max_features=C.MAX_FEATURES, seed=C.SEED).fit(
        R, y, groups=coh["patient_id"].values, icc=icc(R.index, list(R.columns)))
    X = sel.transform(R)
    est, prm, sc = ml.tune_classifier("LR", X, y, coh["patient_id"].values, C.SEED)
    coef = est.coef_.ravel()
    shap_vals = (X.values - X.values.mean(axis=0)) * coef          # exact SHAP for a linear model (logit scale)
    pd.DataFrame(shap_vals, index=X.index, columns=X.columns).to_csv(C.TABLES / "interp_shap_values.csv")
    X.to_csv(C.TABLES / "interp_shap_features.csv")
    pd.DataFrame({"feature": X.columns, "coef": coef, "mean_abs_shap": np.abs(shap_vals).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(C.TABLES / "interp_final_model.csv", index=False)
    with open(C.MODELS / "final_rad_itpt_lr.pkl", "wb") as f:
        pickle.dump({"selector": sel, "model": est, "params": prm}, f)
    print("final model features:", list(X.columns), prm, flush=True)


# ----------------------------------------------------------------------------------------
def load_cnn(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cnn", C.CODE / "07_cnn.py")
    cnn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn)
    m = cnn.build_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval().to(DEVICE), cnn


def gradcam(model, x, layer):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    x = x.clone().requires_grad_(True)
    out = model(x).squeeze(1)
    model.zero_grad()
    out.sum().backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    return cam.detach(), out.detach()


def blur(x, sigma=8.0):
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g, groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3), groups=3)


def part_b():
    coh = D.load_cohort()
    idx = pd.read_csv(C.PROCESSED / "crops_index.csv")
    preds = pd.read_csv(C.PREDS / "cnn_strict.csv")
    preds = preds[preds["variant"] == "standard"]
    rows, abl, examples = [], [], {}
    for fold, fp in preds.groupby("fold"):
        model, cnn = load_cnn(C.MODELS / f"cnn_strict_{fold}.pt")
        for cid in fp["case_id"]:
            z = np.load(C.INTERIM / "crops" / f"{cid}.npz")
            keep = idx[(idx["case_id"] == cid) & (idx["area_frac_of_max"] >= cnn.MIN_AREA_FRAC)
                       & (idx["area_rank"] < cnn.MAX_SLICES)]["slice_idx"].values
            x = torch.from_numpy(z["img"][keep]).float() / 255.0
            x = F.interpolate(x, size=(cnn.SIZE, cnn.SIZE), mode="bilinear", align_corners=False, antialias=True).to(DEVICE)
            m = torch.from_numpy(z["mask"][keep]).float().unsqueeze(1)
            r = torch.from_numpy(z["ring"][keep]).float().unsqueeze(1)
            m = (F.interpolate(m, size=(cnn.SIZE, cnn.SIZE), mode="nearest") > 0.5).to(DEVICE)
            r = (F.interpolate(r, size=(cnn.SIZE, cnn.SIZE), mode="nearest") > 0.5).to(DEVICE)
            xn = cnn.norm(x)
            cam, logit = gradcam(model, xn, model.layer3)
            cam = F.interpolate(cam, size=(cnn.SIZE, cnn.SIZE), mode="bilinear", align_corners=False)
            tot = cam.sum(dim=(1, 2, 3)) + 1e-8
            e_t = ((cam * m).sum(dim=(1, 2, 3)) / tot).cpu().numpy()
            e_r = ((cam * r).sum(dim=(1, 2, 3)) / tot).cpu().numpy()
            a_t = m.float().mean(dim=(1, 2, 3)).cpu().numpy()
            a_r = r.float().mean(dim=(1, 2, 3)).cpu().numpy()
            y = int(coh.loc[cid, "y"])
            rows.append({"case_id": cid, "fold": fold, "y": y, "energy_tumour": e_t.mean(), "energy_ring": e_r.mean(),
                         "area_tumour": a_t.mean(), "area_ring": a_r.mean()})
            xb = blur(x)
            with torch.no_grad():
                variants = {"full": x, "tumour_blurred": torch.where(m, xb, x), "ring_blurred": torch.where(r, xb, x),
                            "outside_tumour_blurred": torch.where(m, x, xb)}
                for v, xv in variants.items():
                    lv = model(cnn.norm(xv)).squeeze(1).float().cpu().numpy()
                    abl.append({"case_id": cid, "fold": fold, "y": y, "variant": v, "logit": float(lv.mean())})
            if len(examples) < 40:
                k = int(np.argmax(m.float().sum(dim=(1, 2, 3)).cpu().numpy()))
                examples[cid] = (x[k].cpu().numpy(), m[k, 0].cpu().numpy(), r[k, 0].cpu().numpy(), cam[k, 0].cpu().numpy())
        print(f"fold {fold} done", flush=True)
    G = pd.DataFrame(rows)
    G["enrich_tumour"] = G["energy_tumour"] / G["area_tumour"]
    G["enrich_ring"] = G["energy_ring"] / G["area_ring"]
    G.to_csv(C.TABLES / "interp_gradcam_regions.csv", index=False)
    A = pd.DataFrame(abl)
    A.to_csv(C.TABLES / "interp_region_ablation.csv", index=False)
    for v, g in A.groupby("variant"):
        print(f"ablation {v:24s} AUC {roc_auc_score(g['y'], g['logit']):.3f}")
    np.savez_compressed(C.PROCESSED / "gradcam_examples.npz",
                        **{f"{cid}__{k}": arr for cid, t in examples.items()
                           for k, arr in zip(("img", "mask", "ring", "cam"), t)})


if __name__ == "__main__":
    parts = sys.argv[1:] or ["a", "b"]
    if "a" in parts:
        part_a()
    if "b" in parts:
        part_b()
