"""07 - Fine-tuned 2.5-D ResNet-18 (ImageNet initialisation) under strict and leaky protocols.

Usage: python 07_cnn.py <protocol> [<protocol> ...]
  strict      patient-grouped 5-fold CV (S1, repeat 0 splits, identical to 06)
  loco        leave-one-centre-out (S2); also reports AdaBN (label-free test-centre BN re-estimation, audit L7)
  s3          develop on C1, test on C2-C4
  slice10     AUDIT L1: slices split at random into 10 folds (slice-level evaluation)
  tumour5     AUDIT L2: tumours split ignoring patients (5 folds)
For every protocol the validation set (early stopping) is 20% of the TRAINING patients; the test fold
is evaluated each epoch only to log what 'picking the best epoch on the test set' would give (audit L5).

Inputs: data/interim/crops/*.npz, data/processed/crops_index.csv, cohort.csv
Outputs: results/predictions/cnn_<protocol>.csv (tumour level), cnn_<protocol>_slices.csv,
         cnn_<protocol>_epochs.csv, results/models/cnn_<protocol>_<fold>.pt (strict and loco only)
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from bcmri import config as C
from bcmri import ml

SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 7
BATCH = 32
MIN_AREA_FRAC = 0.3
MAX_SLICES = 7
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------------------
def load_slices(coh):
    idx = pd.read_csv(C.PROCESSED / "crops_index.csv")
    keep = idx[(idx["area_frac_of_max"] >= MIN_AREA_FRAC) & (idx["area_rank"] < MAX_SLICES)]
    imgs, rows = [], []
    for cid in coh.index:
        z = np.load(C.INTERIM / "crops" / f"{cid}.npz")
        sl = keep[keep["case_id"] == cid]["slice_idx"].values
        x = torch.from_numpy(z["img"][sl]).float() / 255.0
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
        imgs.append((x * 255).round().to(torch.uint8))
        rows += [{"case_id": cid, "slice_idx": int(s)} for s in sl]
    X = torch.cat(imgs)
    meta = pd.DataFrame(rows)
    meta = meta.join(coh[["y", "centre", "patient_id"]], on="case_id")
    return X, meta


def build_model():
    import timm
    m = timm.create_model("resnet18.tv_in1k", pretrained=True, num_classes=1,
                          pretrained_cfg_overlay=dict(file=str(C.WEIGHTS / "timm_resnet18.tv_in1k" / "model.safetensors")))
    return m


def augment(x, gen):
    """x: float (B,3,H,W) in [0,1]. Flip, small affine, intensity jitter -- all on device."""
    b = x.shape[0]
    flip = torch.rand(b, generator=gen) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1).to(x.device), x.flip(-1), x)
    ang = (torch.rand(b, generator=gen) * 30 - 15) * math.pi / 180
    sc = 1 + (torch.rand(b, generator=gen) * 0.2 - 0.1)
    tx = torch.rand(b, generator=gen) * 0.1 - 0.05
    ty = torch.rand(b, generator=gen) * 0.1 - 0.05
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = (0.8 + torch.rand(b, generator=gen) * 0.4).view(-1, 1, 1, 1).to(x.device)
    gain = (0.9 + torch.rand(b, generator=gen) * 0.2).view(-1, 1, 1, 1).to(x.device)
    return (x.clamp(0, 1) ** gamma * gain).clamp(0, 1)


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, X, bs=128):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEVICE).float() / 255.0
        out.append(model(norm(x)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def adabn(model, X, bs=64):
    """Re-estimate BatchNorm statistics on unlabeled target images (AdaBN)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                      # cumulative average
    model.train()
    for i in range(0, len(X), bs):
        model(norm(X[i:i + bs].to(DEVICE).float() / 255.0))
    model.eval()


def tumour_level(meta_sub, logits):
    d = meta_sub[["case_id", "y"]].copy()
    d["logit"] = logits
    t = d.groupby("case_id").agg(y=("y", "first"), logit=("logit", "mean"))
    t["p"] = 1 / (1 + np.exp(-t["logit"]))
    return t


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def train_one(X, meta, tr_idx, va_idx, te_idx, seed, log_rows, tag, level="tumour"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_model().to(DEVICE)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr_idx) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=MAX_EPOCHS * steps_per_epoch,
                                                pct_start=0.1)
    ytr = torch.tensor(meta["y"].values[tr_idx], dtype=torch.float32)
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(1.0, ytr.sum())], device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(tr_idx), generator=gen).numpy()
        tl = 0.0
        for i in range(0, len(perm), BATCH):
            bi = tr_idx[perm[i:i + BATCH]]
            if len(bi) < 2:
                continue
            x = augment(X[bi].to(DEVICE).float() / 255.0, gen)
            yb = ytr[perm[i:i + BATCH]].to(DEVICE)
            loss = lossf(model(norm(x)).squeeze(1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss) * len(bi)
        lv = predict(model, X[va_idx])
        lt = predict(model, X[te_idx])
        if level == "tumour":
            tv, tt = tumour_level(meta.iloc[va_idx], lv), tumour_level(meta.iloc[te_idx], lt)
            auc_v, auc_t = safe_auc(tv["y"], tv["logit"]), safe_auc(tt["y"], tt["logit"])
        else:
            auc_v = safe_auc(meta["y"].values[va_idx], lv)
            auc_t = safe_auc(meta["y"].values[te_idx], lt)
        log_rows.append({**tag, "epoch": ep, "train_loss": tl / len(tr_idx), "val_auc": auc_v, "test_auc": auc_t})
        if auc_v > best + 1e-4:
            best, bad = auc_v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best


def val_split(meta, tr_idx, seed, level):
    """20% of training patients (or slices, for the slice-level audit) for early stopping."""
    y = meta["y"].values[tr_idx]
    if level == "slice":
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        a, b = next(kf.split(tr_idx, y))
        return tr_idx[a], tr_idx[b]
    g = meta["patient_id"].values[tr_idx]
    strata = (meta["centre"].astype(str) + "_" + meta["y"].astype(str)).values[tr_idx]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    a, b = next(sgkf.split(tr_idx, strata, g))
    return tr_idx[a], tr_idx[b]


def main():
    protos = sys.argv[1:] or ["strict", "loco", "s3"]
    coh = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    coh["y"] = coh["label"].astype(int)
    X, meta = load_slices(coh)
    print(f"device {DEVICE}; slices {tuple(X.shape)}; tumours {meta['case_id'].nunique()}", flush=True)
    case_pos = {cid: np.where(meta["case_id"].values == cid)[0] for cid in coh.index}

    def slices_of(case_ids):
        return np.concatenate([case_pos[c] for c in case_ids])

    for proto in protos:
        t0 = time.time()
        level = "slice" if proto == "slice10" else "tumour"
        cases = coh.reset_index()
        if proto == "strict":
            splits = [(f"r0f{f}", tr, te) for r, f, tr, te in ml.strict_cv_splits(cases, C.OUTER_FOLDS, 1, C.SEED)]
        elif proto == "loco":
            splits = [(c, tr, te) for c, tr, te in ml.loco_splits(cases, C.CENTRES)]
        elif proto == "s3":
            splits = [("C1->C2-4", np.where(cases["centre"] == "C1")[0], np.where(cases["centre"] != "C1")[0])]
        elif proto == "tumour5":
            splits = [(f"r0f{f}", tr, te) for r, f, tr, te in ml.tumour_cv_splits(cases, 5, 1, C.SEED)]
        elif proto == "slice10":
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=C.SEED)
            splits = [(f"f{f}", tr, te) for f, (tr, te) in enumerate(kf.split(np.zeros(len(meta)), meta["y"].values))]
        else:
            raise ValueError(proto)
        tum_rows, slice_rows, ep_rows = [], [], []
        for k, (fold, tr, te) in enumerate(splits):
            if level == "tumour":
                tr_s, te_s = slices_of(cases["case_id"].values[tr]), slices_of(cases["case_id"].values[te])
            else:
                tr_s, te_s = tr, te
            fit_s, va_s = val_split(meta, tr_s, C.SEED + k, level)
            tag = {"protocol": proto, "fold": fold}
            model, best_ep, best_val = train_one(X, meta, fit_s, va_s, te_s, C.SEED + k, ep_rows, tag, level)
            lt = predict(model, X[te_s])
            sd = meta.iloc[te_s][["case_id", "slice_idx", "y", "centre"]].copy()
            sd["logit"], sd["fold"], sd["variant"] = lt, fold, "standard"
            slice_rows.append(sd)
            tt = tumour_level(meta.iloc[te_s], lt).reset_index()
            tt["fold"], tt["variant"], tt["best_epoch"], tt["val_auc"] = fold, "standard", best_ep, best_val
            tum_rows.append(tt)
            if proto in ("strict", "loco"):
                torch.save(model.state_dict(), C.MODELS / f"cnn_{proto}_{fold}.pt")
            if proto == "loco":
                adabn(model, X[te_s])
                la = predict(model, X[te_s])
                ta = tumour_level(meta.iloc[te_s], la).reset_index()
                ta["fold"], ta["variant"] = fold, "AdaBN"
                tum_rows.append(ta)
            msg = f"[{proto}] {fold}: best epoch {best_ep}, val AUC {best_val:.3f}"
            if level == "tumour" and tt["y"].nunique() == 2:
                msg += f", test AUC {roc_auc_score(tt['y'], tt['p']):.3f}"
            else:
                msg += f", test slice AUC {safe_auc(sd['y'], sd['logit']):.3f}"
            print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(tum_rows).to_csv(C.PREDS / f"cnn_{proto}.csv", index=False)
        pd.concat(slice_rows).to_csv(C.PREDS / f"cnn_{proto}_slices.csv", index=False)
        pd.DataFrame(ep_rows).to_csv(C.PREDS / f"cnn_{proto}_epochs.csv", index=False)
        print(f"[{proto}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
