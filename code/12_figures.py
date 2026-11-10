"""12 - Main figures (Python/matplotlib; Arial 7 pt; 174-mm double-column width).

Usage: python 12_figures.py [fig1 fig2 ...]   (default: all available)
Every figure writes SVG + PDF + 600-dpi TIFF + PNG preview to results/figures/ and its plotted
numbers to results/figures/source_data/<figure>_*.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter

from bcmri import config as C
from bcmri import figstyle as FS

SRC = C.FIGURES / "source_data"
SRC.mkdir(parents=True, exist_ok=True)


def spread(n, width=0.14):
    """Deterministic horizontal offsets for strip plots (golden-ratio sequence; display only)."""
    return ((np.arange(n) * 0.6180339887) % 1 - 0.5) * width


def cohort():
    coh = pd.read_csv(C.PROCESSED / "cohort.csv").set_index("case_id")
    coh["y"] = coh["label"].astype(int)
    return coh


# ======================================================================================
# Figure 1 - study design and cohort
# ======================================================================================
def _box(ax, x, y, w, h, title, lines, fc, ec, bold_idx=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.7, transform=ax.transAxes, clip_on=False))
    ax.text(x + 0.012, y + h - 0.035, title, transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="top", ha="left")
    step = (h - 0.14) / max(len(lines) - 1, 1) if len(lines) > 1 else 0
    step = min(step, 0.092)
    for i, t in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.135 - i * step, t, transform=ax.transAxes, fontsize=6,
                va="top", ha="left", fontweight="bold" if i in bold_idx else "normal")


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color="#4D4D4D", clip_on=False))


def fig1():
    coh = cohort()
    fig = plt.figure(figsize=(FS.FULL_W, 108 * FS.MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    n_pat, n_tum, n_mibc = coh["patient_id"].nunique(), len(coh), int(coh["y"].sum())
    W, H, Y = 0.188, 0.88, 0.06
    xs = [0.0, 0.2035, 0.407, 0.6105, 0.8135]
    _box(ax, xs[0], Y, W, H, "Public data (FedBCa)",
         ["4 centres, 11/2019–07/2022", "Released and analysed:", f"  {n_tum} tumours, {n_pat} patients",
          "  (no exclusions)", f"{n_mibc} MIBC ({100 * n_mibc / n_tum:.0f}%) by pathology",
          "Axial T2WI + manual masks", "Grouping unit: patient"], "#F2F2F2", "#7F7F7F")
    _box(ax, xs[1], Y, W, H, "Pre-processing",
         ["N4 bias-field correction", "0.7-mm in-plane resampling", "Slice spacing unchanged",
          "3-mm peritumoural ring", "Eroded / dilated masks", "  (ICC robustness)", "2.5D tumour crops"],
         "#F2F2F2", "#7F7F7F")
    _box(ax, xs[2], Y, W, H, "Model families",
         ["Radiomics (IT+PT), L2-LR", "Fine-tuned 2.5D ResNet-18", "Deep-radiomics fusion",
          "Tumour-size baseline", "Frozen features (×3) + probe", "7 other classifiers",
          "Centre-prevalence model"], "#E8F1FA", FS.MODEL_COL["RAD"], bold_idx=(0, 1, 2, 3))
    _box(ax, xs[3], Y, W, H, "Strict evaluation",
         ["S1 patient-grouped 5×5 CV", "S2 leave-one-centre-out", "  (primary endpoint)",
          "S3 C1 → C2–C4", "All fitting inside training", "Cluster-bootstrap 95% CI",
          "Random-effects pooling"], "#E8F1FA", FS.STRICT_COL)
    _box(ax, xs[4], Y, W + 0.004, H, "Leakage audit",
         ["L1 slice-level split", "L2 patients ignored", "L3 selection on all data", "L4 SMOTE before split",
          "L5 best-of-many on test", "L6 threshold on test", "L7 transductive adaptation",
          "L8 centre confounding"], "#FBEAEA", FS.LEAK_COL)
    for i in range(3):
        _arrow(ax, xs[i] + W + 0.002, xs[i + 1] - 0.002, Y + H / 2)
    ax.plot([xs[3] + W + 0.002, xs[4] - 0.002], [Y + H / 2, Y + H / 2], color="#4D4D4D", lw=0.8,
            ls=(0, (2, 1.5)), transform=ax.transAxes, clip_on=False)
    FS.panel_label(ax, "a", dx_pt=-2, dy_pt=0)

    # ---- b: tumours per centre by label
    axb = fig.add_subplot(gs[1, 0])
    t = coh.groupby(["centre", "y"]).size().unstack(fill_value=0).reindex(C.CENTRES)
    pts = coh.groupby("centre")["patient_id"].nunique().reindex(C.CENTRES)
    x = np.arange(len(C.CENTRES))
    axb.bar(x, t[0], 0.62, color=FS.LABEL_COL[0], label="NMIBC")
    axb.bar(x, t[1], 0.62, bottom=t[0], color=FS.LABEL_COL[1], label="MIBC")
    for i, c in enumerate(C.CENTRES):
        tot = t.loc[c].sum()
        axb.text(i, tot + 3, f"{100 * t.loc[c, 1] / tot:.0f}% MIBC", ha="center", va="bottom", fontsize=6)
    axb.set_xticks(x, [f"{c}\n{t.loc[c].sum()} / {pts[c]}" for c in C.CENTRES])
    axb.set_ylabel("Tumours (n)")
    axb.set_xlabel("Centre (tumours / patients)")
    axb.set_ylim(0, 185)
    axb.legend(loc="upper right", handlelength=1, handleheight=0.8, borderaxespad=0.2)
    FS.panel_label(axb, "b")
    t.assign(patients=pts).to_csv(SRC / "fig1b_tumours_by_centre.csv")

    # ---- c: tumour volume by centre and label
    axc = fig.add_subplot(gs[1, 1], sharey=None)
    rows = []
    for i, c in enumerate(C.CENTRES):
        for lab, off in ((0, -0.17), (1, 0.17)):
            v = coh.loc[(coh["centre"] == c) & (coh["y"] == lab), "tumour_ml"].values
            assert (v > 0).all()                       # log axis: volumes are strictly positive
            jit = spread(len(v))
            axc.scatter(i + off + jit, v, s=4, color=FS.LABEL_COL[lab], alpha=0.75, lw=0, rasterized=False)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            axc.plot([i + off - 0.11, i + off + 0.11], [med, med], color="black", lw=1.0)
            axc.plot([i + off, i + off], [q1, q3], color="black", lw=0.6)
            rows.append({"centre": c, "label": "MIBC" if lab else "NMIBC", "n": len(v), "median_ml": med,
                         "q1_ml": q1, "q3_ml": q3})
    axc.set_yscale("log")
    axc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axc.set_xticks(x, C.CENTRES)
    axc.set_xlabel("Centre")
    axc.set_ylabel("Tumour volume (ml)")
    axc.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=FS.LABEL_COL[0], label="NMIBC"),
                        Line2D([], [], marker="o", ls="", ms=3, color=FS.LABEL_COL[1], label="MIBC"),
                        Line2D([], [], color="black", lw=1.0, label="Median (IQR)")],
               loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    axc.set_ylim(0.02, 3000)
    FS.panel_label(axc, "c")
    pd.DataFrame(rows).to_csv(SRC / "fig1c_volume_by_centre_label.csv", index=False)
    coh[["centre", "y", "tumour_ml"]].to_csv(SRC / "fig1c_volume_per_tumour.csv")
    FS.save(fig, C.FIGURES / "Fig1_design_cohort", exclude_axes=[ax])


# ======================================================================================
# shared helpers for result figures
# ======================================================================================
NAMES = {("RAD_ITPT", "LR"): "Radiomics (IT+PT)", ("RAD_IT", "LR"): "Radiomics (IT only)",
         ("CNN", "ResNet18"): "Fine-tuned CNN", ("DLR", "LR"): "Deep-radiomics fusion",
         ("SIZE", "LR"): "Tumour size", ("DL_ResNet50_IN", "LR"): "ResNet-50 probe",
         ("DL_DINOv2_B", "LR"): "DINOv2 probe", ("DL_BiomedCLIP", "LR"): "BiomedCLIP probe",
         ("LATE", "mean"): "Late fusion", ("CENTRE", "prevalence"): "Centre prevalence",
         ("CNN_AdaBN", "ResNet18"): "CNN + AdaBN"}
PRIMARY = [("RAD_ITPT", "LR"), ("CNN", "ResNet18"), ("DLR", "LR"), ("SIZE", "LR")]
SECONDARY = [("RAD_IT", "LR"), ("DL_ResNet50_IN", "LR"), ("DL_DINOv2_B", "LR"), ("DL_BiomedCLIP", "LR"),
             ("LATE", "mean")]
PROTO_NAME = {"S1": "S1 grouped CV", "S2": "S2 leave-one-centre-out", "S3": "S3 C1 → C2–C4"}
PROTO_COL = {"S1": "#7F7F7F", "S2": "#0072B2", "S3": "#E69F00"}


def colour(key):
    m = key[0]
    if m.startswith("RAD"):
        return FS.MODEL_COL["RAD"]
    if m.startswith("CNN"):
        return FS.MODEL_COL["CNN"]
    if m.startswith("DL_"):
        return FS.MODEL_COL["PROBE"]
    return FS.MODEL_COL.get(m, "#4D4D4D")


def roc_points(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def preds(proto):
    """Long table of tumour-level test predictions (all models, repeats) for one strict protocol."""
    coh = cohort()
    d = pd.read_csv(C.PREDS / f"strict_{proto}.csv")
    cnn_name = {"S1": "strict", "S2": "loco", "S3": "s3"}[proto]
    f = C.PREDS / f"cnn_{cnn_name}.csv"
    if f.exists():
        c = pd.read_csv(f)
        c = c[c["variant"] == "standard"].drop(columns=["y"], errors="ignore").join(
            coh[["y", "centre", "patient_id"]], on="case_id")
        c["model"], c["clf"], c["repeat"] = "CNN", "ResNet18", 0
        d = pd.concat([d, c[["model", "clf", "repeat", "fold", "case_id", "p", "y", "centre", "patient_id"]]])
    return d


def fmt_ci(a, lo, hi):
    return f"{a:.2f} ({lo:.2f}–{hi:.2f})"


# ======================================================================================
# Figure 2 - strict, leakage-free performance
# ======================================================================================
def fig2():
    P = pd.read_csv(C.TABLES / "perf_strict.csv")
    BC = pd.read_csv(C.TABLES / "perf_by_centre.csv")
    RE = pd.read_csv(C.TABLES / "perf_random_effects.csv")
    CMP = pd.read_csv(C.TABLES / "compare_vs_size.csv")
    fig = plt.figure(figsize=(FS.FULL_W, 150 * FS.MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.05])

    # ---- a: pooled LOCO ROC curves, primary models
    ax = fig.add_subplot(gs[0, 0])
    d2 = preds("S2")
    src = []
    for key in PRIMARY:
        g = d2[(d2["model"] == key[0]) & (d2["clf"] == key[1])]
        if g.empty:
            continue
        fpr, tpr = roc_points(g["y"].values, g["p"].values)
        r = P[(P["protocol"] == "S2") & (P["model"] == key[0]) & (P["clf"] == key[1])].iloc[0]
        ax.plot(fpr, tpr, color=colour(key), lw=1.3 if key != ("SIZE", "LR") else 1.0,
                ls="-" if key != ("SIZE", "LR") else (0, (3, 1.5)),
                label=f"{NAMES[key]}  {r.auc:.2f}")
        src.append(pd.DataFrame({"model": NAMES[key], "fpr": fpr, "tpr": tpr}))
    ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_anchor("N")           # square ROC panel: keep its top on the shared grid line with panel b
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", handlelength=1.6, fontsize=6.5, borderaxespad=0.1)
    FS.panel_label(ax, "a")
    pd.concat(src).to_csv(SRC / "fig2a_roc_S2.csv", index=False)

    # ---- b: per-centre LOCO forest plot with random-effects summary
    XL2B = (0.3, 1.0)
    ax = fig.add_subplot(gs[:, 1])
    rows, ypos, y0 = [], [], 0
    ticks, ticklabels = [], []
    for key in PRIMARY:
        b = BC[(BC["protocol"] == "S2") & (BC["model"] == key[0]) & (BC["clf"] == key[1])]
        re = RE[(RE["model"] == key[0]) & (RE["clf"] == key[1])]
        if b.empty:
            continue
        ax.text(XL2B[0] + 0.005, y0 - 0.15, NAMES[key], fontsize=7, fontweight="bold", color=colour(key), va="bottom")
        y0 -= 1
        block_top = y0 + 0.45
        for _, r in b.iterrows():
            ax.plot([r.auc_lo, r.auc_hi], [y0, y0], color=colour(key), lw=0.9)
            ax.plot(r.auc, y0, "s", ms=2.2 + 2.5 * r.n / 160, color=colour(key), mec="none")
            ticks.append(y0)
            ticklabels.append(f"{r.centre} (n = {r.n}, {r.n_mibc} MIBC)")
            rows.append({"model": NAMES[key], "row": r.centre, "auc": r.auc, "lo": r.auc_lo, "hi": r.auc_hi})
            y0 -= 0.8
        if not re.empty:
            r = re.iloc[0]
            lo_c, hi_c = max(r.pi_lo, XL2B[0]), min(r.pi_hi, XL2B[1])
            ax.plot([lo_c, hi_c], [y0, y0], color=colour(key), lw=0.6, ls=(0, (1, 1)))
            for edge, mk, out in ((XL2B[0], "<", r.pi_lo < XL2B[0]), (XL2B[1], ">", r.pi_hi > XL2B[1])):
                if out:       # interval continues beyond the axis: mark it instead of clipping silently
                    ax.plot(edge, y0, marker=mk, ms=3, color=colour(key), mec="none", clip_on=False)
            ax.fill([r.lo, r.auc, r.hi, r.auc], [y0, y0 + 0.28, y0, y0 - 0.28], color=colour(key), lw=0)
            ticks.append(y0)
            ticklabels.append(f"Pooled (I² = {100 * r.I2:.0f}%)")
            ax.vlines(0.5, y0 - 0.45, block_top, color="#BDBDBD", lw=0.6, zorder=0)
            rows.append({"model": NAMES[key], "row": "pooled", "auc": r.auc, "lo": r.lo, "hi": r.hi,
                         "pi_lo": r.pi_lo, "pi_hi": r.pi_hi, "I2": r.I2, "tau2": r.tau2})
            y0 -= 1.15
    ax.set_yticks(ticks, ticklabels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*XL2B)
    ax.set_ylim(y0 + 0.4, 0.9)
    ax.set_xlabel("AUC in held-out centre (95% CI)")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color="#4D4D4D", ms=3, label="Held-out centre"),
                       Patch(color="#4D4D4D", label="Random-effects pooled (HKSJ 95% CI)"),
                       Line2D([], [], color="#4D4D4D", lw=0.6, ls=(0, (1, 1)), label="95% prediction interval")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.3, handlelength=1.4)
    FS.panel_label(ax, "b", dx_pt=-90)
    pd.DataFrame(rows).to_csv(SRC / "fig2b_forest_S2.csv", index=False)

    # ---- c: AUC under the three strict protocols
    ax = fig.add_subplot(gs[1, 0])
    keys = PRIMARY + SECONDARY
    rows = []
    for i, key in enumerate(keys):
        for j, proto in enumerate(("S1", "S2", "S3")):
            r = P[(P["protocol"] == proto) & (P["model"] == key[0]) & (P["clf"] == key[1])]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = i + (j - 1) * 0.24
            ax.plot([r.auc_lo, r.auc_hi], [yy, yy], color=PROTO_COL[proto], lw=0.8)
            ax.plot(r.auc, yy, "o", ms=2.6, color=PROTO_COL[proto], mec="none")
            rows.append({"model": NAMES[key], "protocol": proto, "auc": r.auc, "lo": r.auc_lo, "hi": r.auc_hi})
    ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
    for t, k in zip(ax.get_yticklabels(), keys):
        t.set_fontweight("bold" if k in PRIMARY else "normal")
    ax.set_ylim(len(keys) - 1 + 2.3, -0.6)        # inverted; band below the last row holds the legend
    ax.vlines(0.5, -0.6, len(keys) - 1 + 0.5, color="#BDBDBD", lw=0.6, zorder=0)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC (95% CI)")
    ax.legend(handles=[Line2D([], [], marker="o", color=PROTO_COL[p], ms=2.6, lw=0.8, label=PROTO_NAME[p])
                       for p in ("S1", "S2", "S3")], loc="lower right", borderaxespad=0.1, handlelength=1.4)
    FS.panel_label(ax, "c")
    pd.DataFrame(rows).to_csv(SRC / "fig2c_auc_by_protocol.csv", index=False)
    FS.save(fig, C.FIGURES / "Fig2_strict_performance")
    # paired differences versus tumour size are reported in Table 2 / Supplementary Table
    CMP.to_csv(SRC / "fig2_compare_vs_size.csv", index=False)


# ======================================================================================
# Figure 4 - centre shortcut: image features encode the acquiring centre
# ======================================================================================
def _tsne_panel(ax, E, name, title):
    e = E[E["feature_set"] == name]
    for c in C.CENTRES:
        for lab, mk in ((0, "o"), (1, "^")):
            g = e[(e["centre"] == c) & (e["label"] == lab)]
            ax.scatter(g["x"], g["y_"], s=7 if lab else 6, marker=mk, lw=0.5,
                       facecolor=FS.CENTRE_COL[c] if lab else "none", edgecolor=FS.CENTRE_COL[c], alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title, loc="left", fontsize=7)


def fig4():
    E = pd.read_csv(C.TABLES / "domain_embeddings.csv")
    I = pd.read_csv(C.TABLES / "domain_identifiability.csv")
    P = pd.read_csv(C.TABLES / "perf_strict.csv")
    fig = plt.figure(figsize=(FS.FULL_W, 128 * FS.MM), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    _tsne_panel(ax, E, "Radiomics", "Radiomics features (IT+PT)")
    handles = [Line2D([], [], marker="s", ls="", color=FS.CENTRE_COL[c], ms=3.5, label=c) for c in C.CENTRES]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec="#4D4D4D", ms=3, mew=0.6, label="NMIBC"),
                Line2D([], [], marker="^", ls="", color="#4D4D4D", ms=3, label="MIBC")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2,
              handletextpad=0.2, labelspacing=0.35)
    FS.panel_label(ax, "a", dx_pt=-12)
    ax = fig.add_subplot(gs[0, 1])
    _tsne_panel(ax, E, "DINOv2_B", "Frozen DINOv2 features")
    FS.panel_label(ax, "b", dx_pt=-12)
    E.to_csv(SRC / "fig4ab_tsne.csv", index=False)

    # ---- c: centre identifiability
    ax = fig.add_subplot(gs[1, 0])
    lab = {"Radiomics (IT+PT, all)": "Radiomics, all", "Radiomics, first-order only": "Radiomics, first-order",
           "Radiomics, shape only": "Radiomics, shape", "Tumour size (2 features)": "Tumour size",
           "Deep: ResNet50_IN": "ResNet-50 features", "Deep: DINOv2_B": "DINOv2 features",
           "Deep: BiomedCLIP": "BiomedCLIP features"}
    I = I.assign(name=I["feature_set"].map(lab)).sort_values("macro_auc_ovr")
    yy = np.arange(len(I))
    ax.hlines(yy, 0.5, I["macro_auc_ovr"], color="#BDBDBD", lw=0.8)
    ax.plot(I["macro_auc_ovr"], yy, "o", ms=3.5, color="#4D4D4D")
    for k, (_, r) in enumerate(I.iterrows()):
        ax.annotate(f"{r.macro_auc_ovr:.2f} ({r.accuracy:.2f})", xy=(r.macro_auc_ovr, k), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=6)
    ax.set_yticks(yy, I["name"])
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="#BDBDBD", lw=0.6)
    ax.set_xlabel("Centre identification, macro AUC")
    FS.panel_label(ax, "c", dx_pt=-12)
    I.to_csv(SRC / "fig4c_identifiability.csv", index=False)

    # ---- d: pooled versus within-centre discrimination in pooled CV (S1)
    ax = fig.add_subplot(gs[1, 1])
    keys = PRIMARY + [("CENTRE", "prevalence")]
    rows = []
    for k, key in enumerate(keys):
        r = P[(P["protocol"] == "S1") & (P["model"] == key[0]) & (P["clf"] == key[1])]
        if r.empty:
            continue
        r = r.iloc[0]
        w = r.get("auc_within_centre_weighted", np.nan)
        ax.plot([w, r.auc], [k, k], color="#BDBDBD", lw=1.0, zorder=0)
        ax.plot(r.auc, k, "o", ms=4, color=colour(key))
        ax.plot(w, k, "o", ms=4, mfc="white", mec=colour(key), mew=0.9)
        rows.append({"model": NAMES[key], "pooled_auc": r.auc, "within_centre_auc": w})
    ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.axvline(0.5, color="#BDBDBD", lw=0.6, zorder=0)
    ax.set_xlabel("AUC in pooled cross-validation (S1)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#4D4D4D", ms=4, label="Pooled across centres"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec="#4D4D4D", ms=4,
                              label="Within centre (weighted mean)")],
              loc="center right", bbox_to_anchor=(1.0, len(keys) - 1.5), bbox_transform=ax.get_yaxis_transform(),
              borderaxespad=0.1)          # in the empty band between the last two rows, clear of all data
    FS.panel_label(ax, "d", dx_pt=-12)
    pd.DataFrame(rows).to_csv(SRC / "fig4d_pooled_vs_within.csv", index=False)
    FS.save(fig, C.FIGURES / "Fig4_centre_shortcut")


# ======================================================================================
# Figure 3 - leakage audit
# ======================================================================================
PUBLISHED = [(0.866, "Dataset paper (centralised ResNet-50)"), (0.991, "Highest published (DADCNet)")]
FOLD_MEAN_REF = {"L5 Epoch picked on test†": "Epoch chosen on validation (reference)†",
                 "L1 + L5 epoch picked on test†": "L1, epoch chosen on validation†",
                 "L5 LOCO, epoch picked on held-out centre†": "LOCO, epoch chosen on validation (reference)†"}


def _audit_rows(A):
    """Ordered display rows: (section, display label, audit variant prefix, is_reference)."""
    spec = [
        ("Radiomics, pooled CV (S1)", [
            ("Strict (reference)", "Strict (patient-grouped CV)", True),
            ("L2 Patients ignored", "L2 Tumour-level split", False),
            ("L3 Selection on all data", "L3 Feature selection on all data", False),
            ("L4 SMOTE before split*", "L4 SMOTE before split", False),
            ("L5 Best of all configurations", "L5 Best of", False),
            ("L7 ComBat on all data", "L7 ComBat on all data, label-free", False),
            ("L7 ComBat on all data + label", "L7 ComBat on all data, label as covariate", False)]),
        ("Radiomics, leave-one-centre-out (S2)", [
            ("No harmonisation (reference)", "LOCO, no harmonisation", True),
            ("ComBat, training centres only", "LOCO, ComBat on training centres only", False),
            ("L7 ComBat incl. test centre", "L7 LOCO, ComBat incl. test centre, label-free", False),
            ("L7 ComBat incl. test centre + label", "L7 LOCO, ComBat incl. test centre, label as covariate", False)]),
        ("Fine-tuned CNN", [
            ("Strict, tumour level (reference)", "Strict (patient-grouped CV), tumour level", True),
            ("L2 Patients ignored", "L2 Tumour-level split", False),
            ("Strict, slice level (reference)", "Strict (patient-grouped CV), slice level", True),
            ("L1 Slice-level split", "L1 Slice-level random split", False),
            ("Epoch chosen on validation (reference)†", "Epoch chosen on validation", True),
            ("L5 Epoch picked on test†", "L5 Epoch chosen on test", False),
            ("L1, epoch chosen on validation†", "L1 Slice-level split, epoch chosen on validation", False),
            ("L1 + L5 epoch picked on test†", "L1+L5 Slice-level split", False),
            ("LOCO (reference)", "LOCO, standard", True),
            ("L7 AdaBN on test-centre images", "L7 LOCO with AdaBN", False),
            ("LOCO, epoch chosen on validation (reference)†", "LOCO, epoch chosen on validation", True),
            ("L5 LOCO, epoch picked on held-out centre†", "L5 LOCO, epoch chosen on held-out centre", False)]),
    ]
    out = []
    for sec, items in spec:
        grp = "CNN" if sec.startswith("Fine") else "Radiomics"
        for disp, prefix, ref in items:
            r = A[(A["group"] == grp) & A["variant"].astype(str).str.startswith(prefix)]
            if r.empty:
                continue
            out.append((sec, disp, r.iloc[0], ref))
    return out


def fig3():
    A = pd.read_csv(C.TABLES / "audit_summary.csv")
    L5 = pd.read_csv(C.TABLES / "audit_L5_configs.csv")
    rows = _audit_rows(A)
    fig = plt.figure(figsize=(FS.FULL_W, 160 * FS.MM), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1])
    top = outer[0].subgridspec(1, 2, width_ratios=[2.2, 1])
    bot = outer[1].subgridspec(1, 2)
    ax, axd = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    for a_, gid in ((ax, "a"), (axd, "b")):
        a_.set_gid(gid)
    ypos, labels, y, prev_sec, src = [], [], 0.0, None, []
    sec_y = []
    for sec, disp, r, ref in rows:
        if sec != prev_sec:
            y += 0.9 if prev_sec else 0.0
            sec_y.append((y, sec))
            y += 1.0
            prev_sec = sec
        col = FS.STRICT_COL if ref else FS.LEAK_COL
        if pd.notna(r.get("lo")):
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=0.9)
        ax.plot(r.auc, y, "o" if pd.notna(r.get("lo")) else "D", ms=3.2, color=col,
                mfc=col if pd.notna(r.get("lo")) else "white", mew=0.9)
        if pd.notna(r.get("delta")):
            axd.plot([r.delta_lo, r.delta_hi], [y, y], color=col, lw=0.9)
            axd.plot(r.delta, y, "o", ms=3.2, color=col)
        elif disp in FOLD_MEAN_REF:          # fold-mean AUCs: plain difference, no CI
            ref_r = next(rr for _, dd, rr, _ in rows if dd == FOLD_MEAN_REF[disp])
            axd.plot(r.auc - ref_r.auc, y, "D", ms=3.2, color=col, mfc="white", mew=0.9)
        ypos.append(y)
        labels.append(disp)
        src.append({"section": sec, "row": disp, "variant": r.variant, "auc": r.auc, "lo": r.get("lo"),
                    "hi": r.get("hi"), "delta": r.get("delta"), "delta_lo": r.get("delta_lo"),
                    "delta_hi": r.get("delta_hi"), "unit": r.get("unit")})
        y += 1.0
    for x0, lab in PUBLISHED:
        ax.axvline(x0, color="#7F7F7F", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(ypos, labels)
    for t, (_, _, _, ref) in zip(ax.get_yticklabels(), rows):
        t.set_color(FS.STRICT_COL if ref else "black")
    for yy, sec in sec_y:          # section headers, right-aligned with the tick labels
        ax.annotate(sec, xy=(0, yy), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7, fontweight="bold",
                    annotation_clip=False)
    ax.set_ylim(y - 0.4, -0.6)
    axd.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC (95% CI)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for x0, lab in PUBLISHED:
        ax.annotate(f"{x0:.2f}", xy=(x0, -0.6), xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=6, color="#4D4D4D", annotation_clip=False)
    axd.axvline(0, color="#7F7F7F", lw=0.6, zorder=0)
    axd.set_yticks([])
    axd.spines["left"].set_visible(False)
    axd.set_xlabel("ΔAUC versus reference (95% CI)")
    FS.panel_label(ax, "a", dx_pt=-6)
    FS.panel_label(axd, "b", dx_pt=-6)
    pd.DataFrame(src).to_csv(SRC / "fig3ab_audit.csv", index=False)

    # ---- c: L5 winner's curse across configurations
    axc = fig.add_subplot(bot[0])
    axc.set_gid("c")
    v = L5["auc_mean_2rep"].values
    ref = A[A["variant"].astype(str).str.startswith("Strict, repeats 0-1")]
    span = np.r_[v, ref.iloc[0].auc] if not ref.empty else v
    lo_c = np.floor((span.min() - 0.01) / 0.05) * 0.05       # every configuration and both markers in view
    hi_c = np.ceil((span.max() + 0.01) / 0.05) * 0.05
    cnt, _, _ = axc.hist(v, bins=np.arange(lo_c, hi_c + 1e-9, 0.005), color="#BDBDBD", edgecolor="white", lw=0.3)
    axc.set_ylim(0, 1.3 * cnt.max())                          # headroom: line labels above the tallest bar
    if not ref.empty:
        axc.axvline(ref.iloc[0].auc, color=FS.STRICT_COL, lw=1.0)
        axc.annotate("Pre-specified\nmodel", xy=(ref.iloc[0].auc, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(-3, -2), textcoords="offset points", ha="right", va="top", fontsize=6,
                     color=FS.STRICT_COL)
    axc.axvline(v.max(), color=FS.LEAK_COL, lw=1.0)
    axc.annotate("Best of\n" + f"{len(v)}", xy=(v.max(), 1.0), xycoords=("data", "axes fraction"),
                 xytext=(3, -2), textcoords="offset points", ha="left", va="top", fontsize=6, color=FS.LEAK_COL)
    axc.set_xlim(lo_c, hi_c)
    axc.set_xlabel("Radiomics test AUC (mean of two S1 repeats)")
    axc.set_ylabel("Configurations (n)")
    FS.panel_label(axc, "c")
    L5.to_csv(SRC / "fig3c_L5_configs.csv", index=False)

    # ---- d: CNN test AUC by training epoch (what epoch-picking on the test set exploits)
    axe = fig.add_subplot(bot[1])
    axe.set_gid("d")
    srcd = []
    for proto, col, lab in (("strict", FS.STRICT_COL, "Patient-grouped CV, tumour level"),
                            ("slice10", FS.LEAK_COL, "L1 slice-level split, slice level")):
        f = C.PREDS / f"cnn_{proto}_epochs.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f)
        for _, g in e.groupby("fold"):
            axe.plot(g["epoch"] + 1, g["test_auc"], color=col, lw=0.5, alpha=0.45)
            k = g["val_auc"].idxmax()
            axe.plot(g.loc[k, "epoch"] + 1, g.loc[k, "test_auc"], "o", ms=2.6, color=col, mec="none")
            j = g["test_auc"].idxmax()
            axe.plot(g.loc[j, "epoch"] + 1, g.loc[j, "test_auc"], "*", ms=4.5, color=col, mec="none")
        e["protocol"] = proto
        srcd.append(e)
        axe.plot([], [], color=col, lw=1.0, label=lab)
    axe.plot([], [], "o", ms=2.6, color="#4D4D4D", label="Epoch chosen on validation")
    axe.plot([], [], "*", ms=4.5, color="#4D4D4D", label="Epoch with best test AUC")
    axe.set_xlabel("Training epoch")
    axe.set_ylabel("Test AUC")
    lo_d = min([0.5] + [np.floor(x["test_auc"].min() / 0.05) * 0.05 for x in srcd])   # keep every epoch visible
    axe.set_ylim(lo_d, 1.0)
    axe.legend(loc="lower right", borderaxespad=0.1, handlelength=1.4)
    FS.panel_label(axe, "d")
    if srcd:
        pd.concat(srcd).to_csv(SRC / "fig3d_cnn_epochs.csv", index=False)
    FS.save(fig, C.FIGURES / "Fig3_leakage_audit", row_groups=[["a", "b"], ["c", "d"]])


# ======================================================================================
# Figure 5 - what the models use
# ======================================================================================
_CLS = {"firstorder": "FO", "glcm": "GLCM", "glrlm": "GLRLM", "glszm": "GLSZM", "gldm": "GLDM",
        "ngtdm": "NGTDM", "shape": "Shape"}
_FEAT = {"SmallAreaLowGrayLevelEmphasis": "SALGLE", "SmallAreaHighGrayLevelEmphasis": "SAHGLE",
         "LargeAreaLowGrayLevelEmphasis": "LALGLE", "LargeAreaHighGrayLevelEmphasis": "LAHGLE",
         "SmallDependenceLowGrayLevelEmphasis": "SDLGLE", "SmallDependenceHighGrayLevelEmphasis": "SDHGLE",
         "LargeDependenceLowGrayLevelEmphasis": "LDLGLE", "LargeDependenceHighGrayLevelEmphasis": "LDHGLE",
         "ShortRunLowGrayLevelEmphasis": "SRLGLE", "ShortRunHighGrayLevelEmphasis": "SRHGLE",
         "LongRunLowGrayLevelEmphasis": "LRLGLE", "LongRunHighGrayLevelEmphasis": "LRHGLE",
         "GrayLevelNonUniformityNormalized": "GLNN", "GrayLevelNonUniformity": "GLN",
         "RunLengthNonUniformityNormalized": "RLNN", "RunLengthNonUniformity": "RLN",
         "DependenceNonUniformityNormalized": "DNN", "DependenceNonUniformity": "DN",
         "SizeZoneNonUniformityNormalized": "SZNN", "SizeZoneNonUniformity": "SZN",
         "Maximum2DDiameterSlice": "Max 2D diameter (axial)", "InverseVariance": "Inverse variance",
         "InterquartileRange": "IQR", "MeanAbsoluteDeviation": "MAD",
         "RobustMeanAbsoluteDeviation": "rMAD", "LowGrayLevelZoneEmphasis": "LGLZE",
         "HighGrayLevelZoneEmphasis": "HGLZE", "ClusterProminence": "Cluster prominence",
         "ClusterShade": "Cluster shade", "ClusterTendency": "Cluster tendency"}


def pretty_feature(name):
    """'PT_log-sigma-2-0-mm-3D_glcm_Idmn' -> 'PT LoGσ2 GLCM Idmn'; deep PCs -> 'DINOv2 PC3'."""
    if name.startswith("PC"):
        pc, bb = name.split("_", 1)
        return f"{NAMES.get(('DL_' + bb, 'LR'), bb).replace(' probe', '')} {pc}"
    region, rest = name.split("_", 1)
    img, cls, feat = rest.rsplit("_", 2)
    if img.startswith("log-sigma"):
        img = "LoGσ" + img.split("-")[2]
    elif img.startswith("wavelet"):
        img = "wav-" + img.split("-")[1]
    else:
        img = ""
    return " ".join(x for x in (region, img, _CLS.get(cls, cls), _FEAT.get(feat, feat)) if x)


def _contour(ax, m, **kw):
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], **kw)


def fig5():
    from bcmri import stats as S
    shap = pd.read_csv(C.TABLES / "interp_shap_values.csv", index_col=0)
    feat = pd.read_csv(C.TABLES / "interp_shap_features.csv", index_col=0)
    fm = pd.read_csv(C.TABLES / "interp_final_model.csv")
    freq = pd.read_csv(C.TABLES / "interp_selection_frequency.csv")
    G = pd.read_csv(C.TABLES / "interp_gradcam_regions.csv")
    AB = pd.read_csv(C.TABLES / "interp_region_ablation.csv")
    coh = cohort()
    fig = plt.figure(figsize=(FS.FULL_W, 150 * FS.MM), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.9])
    top = outer[0].subgridspec(1, 2, width_ratios=[1.25, 1])
    bot = outer[1].subgridspec(1, 3, width_ratios=[1.25, 0.8, 0.8])

    # ---- a: SHAP beeswarm (final RAD model refitted on all tumours)
    ax = fig.add_subplot(top[0])
    ax.set_gid("a")
    order = fm["feature"].tolist()[::-1]
    from matplotlib import cm
    cmap = plt.get_cmap("RdBu_r")
    for i, f in enumerate(order):
        sv, fv = shap[f].values, feat[f].values
        z = np.clip((fv - np.median(fv)) / (np.percentile(fv, 95) - np.percentile(fv, 5) + 1e-9) + 0.5, 0, 1)
        o = np.argsort(sv)
        dens = np.zeros(len(sv))
        h, edges = np.histogram(sv, bins=30)
        idx = np.clip(np.digitize(sv, edges) - 1, 0, 29)
        for b in range(30):        # deterministic within-bin spread (beeswarm-like)
            k = np.where(idx == b)[0]
            dens[k] = spread(len(k), width=min(0.8, 0.06 * len(k) ** 0.5 + 0.05))
        ax.scatter(sv, i + dens, c=cmap(z), s=3, lw=0)
    ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=0)
    ax.set_yticks(range(len(order)), [pretty_feature(f) for f in order])
    ax.set_xlabel("SHAP value (log-odds of MIBC)")
    sm = cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Feature value", fontsize=6.5)
    cb.outline.set_linewidth(0.4)
    FS.panel_label(ax, "a", dx_pt=-6)
    shap.to_csv(SRC / "fig5a_shap_values.csv")

    # ---- b: selection stability across the 25 strict folds
    ax = fig.add_subplot(top[1])
    ax.set_gid("b")
    fq = freq.sort_values("fraction", ascending=False).head(12)[::-1]
    ax.barh(range(len(fq)), fq["fraction"], color=FS.MODEL_COL["RAD"], height=0.65)
    ax.set_yticks(range(len(fq)), [pretty_feature(f) for f in fq["feature"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selected in fraction of S1 folds")
    FS.panel_label(ax, "b", dx_pt=-6)
    fq.to_csv(SRC / "fig5b_selection_frequency.csv", index=False)

    # ---- c: Grad-CAM examples (first two MIBC and NMIBC tumours of the first fold; not chosen by accuracy)
    sub = bot[0].subgridspec(2, 2, wspace=0.02, hspace=0.02)
    z = np.load(C.PROCESSED / "gradcam_examples.npz")
    cids = sorted({k.split("__")[0] for k in z.files}, key=lambda c: list(coh.index).index(c))
    pick = [c for c in cids if coh.loc[c, "y"] == 1][:2] + [c for c in cids if coh.loc[c, "y"] == 0][:2]
    img_axes = []
    for j, cid in enumerate(pick):
        a_ = fig.add_subplot(sub[j // 2, j % 2])
        a_.set_anchor("S" if j < 2 else "N")         # square images hug each other, spare height goes outside
        img_axes.append(a_)
        im = z[f"{cid}__img"][1]
        cam = z[f"{cid}__cam"]
        cam = cam / (cam.max() + 1e-8)
        a_.imshow(im, cmap="gray", vmin=0, vmax=1)
        a_.imshow(np.ma.masked_less(cam, 0.2), cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        _contour(a_, z[f"{cid}__mask"], colors="#56B4E9", linewidths=0.6)
        _contour(a_, z[f"{cid}__ring"] | z[f"{cid}__mask"], colors="#F0E442", linewidths=0.4, linestyles="dashed")
        a_.set_xticks([])
        a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
        a_.text(0.03, 0.97, "MIBC" if coh.loc[cid, "y"] else "NMIBC", transform=a_.transAxes, color="white",
                fontsize=6, va="top", ha="left")

    # ---- d: Grad-CAM energy share inside tumour and ring relative to their area share
    ax = fig.add_subplot(bot[1])
    ax.set_gid("d")
    vals = [G["enrich_tumour"].values, G["enrich_ring"].values]
    for k, v in enumerate(vals):
        ax.scatter(k + spread(len(v), 0.35), v, s=3, color=["#56B4E9", "#E69F00"][k], lw=0, alpha=0.7)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([k - 0.22, k + 0.22], [med, med], color="black", lw=1.0)
        ax.plot([k, k], [q1, q3], color="black", lw=0.6)
    ax.axhline(1, color="#7F7F7F", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([0, 1], ["Tumour", "3-mm ring"])
    ax.set_ylabel("Grad-CAM enrichment\n(energy share / area share)")
    FS.panel_label(ax, "d", dx_pt=-6)
    G.to_csv(SRC / "fig5d_gradcam_regions.csv", index=False)

    # ---- e: occlusion of regions (blurred) -> tumour-level AUC
    ax = fig.add_subplot(bot[2])
    ax.set_gid("e")
    order_v = [("full", "Original"), ("ring_blurred", "Ring blurred"), ("tumour_blurred", "Tumour blurred"),
               ("outside_tumour_blurred", "Outside blurred")]
    rows = []
    for k, (v, lab) in enumerate(order_v):
        g = AB[AB["variant"] == v].set_index("case_id")
        b = S.cluster_bootstrap_auc(g["y"].values, g["logit"].values, coh.loc[g.index, "patient_id"].values,
                                    C.N_BOOT, C.SEED)
        ax.plot([b["lo"], b["hi"]], [k, k], color=FS.MODEL_COL["CNN"], lw=0.9)
        ax.plot(b["auc"], k, "o", ms=3.2, color=FS.MODEL_COL["CNN"])
        rows.append({"variant": lab, "auc": b["auc"], "lo": b["lo"], "hi": b["hi"]})
    ax.set_yticks(range(len(order_v)), [lab for _, lab in order_v])
    ax.invert_yaxis()
    ax.axvline(0.5, color="#BDBDBD", lw=0.6, zorder=0)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("CNN AUC (95% CI)")
    FS.panel_label(ax, "e", dx_pt=-6)
    pd.DataFrame(rows).to_csv(SRC / "fig5e_region_ablation.csv", index=False)
    # panel c label: left edge of the image grid, same height as the d and e labels
    fig.canvas.draw()
    ax_d = next(a for a in fig.axes if a.get_gid() == "d")
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(-6 / 72, 4 / 72, fig.dpi_scale_trans)
    fig.text(img_axes[0].get_position().x0, ax_d.get_position().y1, "c", transform=fig.transFigure + off,
             fontsize=8, fontweight="bold", ha="left", va="bottom")
    FS.save(fig, C.FIGURES / "Fig5_interpretability", exclude_axes=img_axes, row_groups=[["a", "b"], ["d", "e"]])


# ======================================================================================
# Figure 6 - calibration and clinical utility under leave-one-centre-out validation
# ======================================================================================
def fig6():
    CAL = pd.read_csv(C.TABLES / "calibration_curves.csv")
    DCA = pd.read_csv(C.TABLES / "dca.csv")
    BC = pd.read_csv(C.TABLES / "perf_by_centre.csv")
    P = pd.read_csv(C.TABLES / "perf_strict.csv")
    fig = plt.figure(figsize=(FS.FULL_W, 62 * FS.MM), layout="constrained")
    gs = fig.add_gridspec(1, 3)
    # ---- a: calibration curves (S2 pooled)
    ax = fig.add_subplot(gs[0])
    for key in PRIMARY:
        c = CAL[(CAL["protocol"] == "S2") & (CAL["model"] == key[0]) & (CAL["clf"] == key[1])]
        if c.empty:
            continue
        r = P[(P["protocol"] == "S2") & (P["model"] == key[0]) & (P["clf"] == key[1])].iloc[0]
        ax.errorbar(c["p_mean"], c["obs"], yerr=1.96 * c["se"], color=colour(key), marker="o", ms=2.6, lw=0.9,
                    elinewidth=0.5, capsize=0, label=NAMES[key])
    ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk of MIBC")
    ax.set_ylabel("Observed proportion of MIBC")
    ax.legend(loc="upper left", borderaxespad=0.1, handlelength=1.2, fontsize=6.5)
    FS.panel_label(ax, "a")
    CAL[CAL["protocol"] == "S2"].to_csv(SRC / "fig6a_calibration_S2.csv", index=False)
    # ---- b: prevalence shift -- mean predicted risk vs observed prevalence in each held-out centre
    ax = fig.add_subplot(gs[1])
    for key in PRIMARY:
        b = BC[(BC["protocol"] == "S2") & (BC["model"] == key[0]) & (BC["clf"] == key[1])]
        ax.plot(b["prevalence"], b["mean_p"], "o", ms=3.2, color=colour(key), mec="white", mew=0.3)
    prev_c = BC[BC["protocol"] == "S2"].groupby("centre")["prevalence"].first().sort_values()
    last_x, row = -1.0, 0
    for c, x in prev_c.items():
        row = 1 - row if x - last_x < 0.06 else 0             # second row when two centres are too close
        ax.annotate(c, xy=(x, 0.03 + 0.06 * row), ha="center", va="bottom", fontsize=6, color="#4D4D4D")
        last_x = x
    ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=0.6, zorder=0)
    b = BC[BC["protocol"] == "S2"]
    hi_b = max(0.8, np.ceil(max(b["prevalence"].max(), b["mean_p"].max()) / 0.2) * 0.2)   # no point off the axes
    ax.set_xlim(0, hi_b)
    ax.set_ylim(0, hi_b)
    ax.set_xticks(np.arange(0, hi_b + 1e-9, 0.2))      # last tick on the edge, like panels a and c (equal gutters)
    ax.set_yticks(np.arange(0, hi_b + 1e-9, 0.2))
    ax.set_xlabel("Observed MIBC prevalence\nin held-out centre")
    ax.set_ylabel("Mean predicted risk")
    FS.panel_label(ax, "b")
    BC[BC["protocol"] == "S2"].to_csv(SRC / "fig6b_prevalence_vs_mean_risk.csv", index=False)
    # ---- c: decision curves (S2)
    ax = fig.add_subplot(gs[2])
    d = DCA[DCA["protocol"] == "S2"]
    for key in PRIMARY:
        g = d[(d["model"] == key[0]) & (d["clf"] == key[1])]
        ax.plot(*cut_below(g["threshold"].values, g["net_benefit"].values, YLO_6C), color=colour(key), lw=1.0,
                ls="-" if key != ("SIZE", "LR") else (0, (3, 1.5)), label=NAMES[key])
    g = d[d["model"] == "Treat all"]
    ax.plot(*cut_below(g["threshold"].values, g["net_benefit"].values, YLO_6C), color="#7F7F7F", lw=0.8,
            label="Treat all as MIBC")
    ax.axhline(0, color="black", lw=0.6, label="Treat none")
    ax.set_xlim(0.05, 0.6)
    ax.set_ylim(YLO_6C, 0.35)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", borderaxespad=0.1, handlelength=1.4)
    FS.panel_label(ax, "c")
    d.to_csv(SRC / "fig6c_dca_S2.csv", index=False)
    FS.save(fig, C.FIGURES / "Fig6_calibration_dca")


YLO_6C = -0.05


def cut_below(x, y, ylo):
    """End a curve where it first falls below the axis limit (interpolated), so no path runs outside the axes."""
    below = np.where(y < ylo)[0]
    if len(below) == 0:
        return x, y
    i = below[0]
    if i == 0:
        return x[:0], y[:0]
    xc = x[i - 1] + (ylo - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    return np.r_[x[:i], xc], np.r_[y[:i], ylo]


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(FIGS)):
        FIGS[name]()
        print(f"{name} done", flush=True)
