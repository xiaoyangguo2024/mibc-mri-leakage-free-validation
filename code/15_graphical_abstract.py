"""15 - Draft graphical abstract (data-driven; Python/matplotlib).

Insights into Imaging asks for a graphical abstract at revision, built on the journal's template. This draft shows
the content and numbers to transfer: design, leakage-free performance on held-out centres, and the effect of single
evaluation flaws. All numbers are read from results/tables (no hand-typed values).
Output: results/figures/Graphical_abstract.{svg,pdf,tiff,png}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from bcmri import config as C
from bcmri import figstyle as FS

W, H = 200 * FS.MM, 112.5 * FS.MM          # 16:9


def main():
    coh = pd.read_csv(C.PROCESSED / "cohort.csv")
    P = pd.read_csv(C.TABLES / "perf_strict.csv")
    A = pd.read_csv(C.TABLES / "audit_summary.csv")
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")

    # ---- title band
    fig.text(0.02, 0.955, "MRI radiomics and deep learning for muscle-invasive bladder cancer:",
             fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.02, 0.905, "how much of the reported accuracy survives leakage-free validation?",
             fontsize=9.5, fontweight="bold", va="top")

    def box(x, y, w, h, title):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                          transform=fig.transFigure, fc="#F5F7FA", ec="#C9D1DB", lw=0.6,
                                          zorder=0))
        fig.text(x + 0.012, y + h - 0.025, title, fontsize=8, fontweight="bold", va="top", color="#1F3A5F")

    # ---- 1: data and design
    box(0.02, 0.18, 0.27, 0.67, "Public four-centre T2WI dataset")
    n_t, n_p, n_m = len(coh), coh["patient_id"].nunique(), int(coh["label"].sum())
    lines = [f"{n_t} tumours, {n_p} patients", f"{n_m} MIBC ({100 * n_m / n_t:.0f}%)", "",
             "Pre-specified models:", "  radiomics, fine-tuned CNN,", "  deep-radiomics fusion",
             "Comparator: tumour size", "", "Leakage-free validation:", "  patient-grouped CV and",
             "  held-out centres (primary)", "", "Leakage audit:", "  one evaluation flaw at a time"]
    for i, t in enumerate(lines):
        fig.text(0.035, 0.775 - i * 0.04, t, fontsize=7, va="top",
                 fontweight="bold" if t.endswith(":") else "normal")

    # ---- 2: held-out centres
    box(0.31, 0.18, 0.325, 0.67, "Held-out centres: no gain over size")
    ax = fig.add_axes([0.43, 0.30, 0.19, 0.44])
    ax.set_zorder(2)                 # above the rounded box drawn behind it
    ax.patch.set_alpha(0)
    keys = [("SIZE", "LR", "Tumour size", FS.MODEL_COL["SIZE"]), ("RAD_ITPT", "LR", "Radiomics", FS.MODEL_COL["RAD"]),
            ("DLR", "LR", "Fusion", FS.MODEL_COL["DLR"]), ("CNN", "ResNet18", "CNN", FS.MODEL_COL["CNN"])]
    for i, (m, c, lab, col) in enumerate(keys):
        r = P[(P["protocol"] == "S2") & (P["model"] == m) & (P["clf"] == c)].iloc[0]
        ax.plot([r.auc_lo, r.auc_hi], [i, i], color=col, lw=1.2)
        ax.plot(r.auc, i, "o", ms=4.5, color=col)
        ax.text(r.auc_lo - 0.015, i, f"{r.auc:.2f}", va="center", ha="right", fontsize=7, color=col)
    size_auc = P[(P["protocol"] == "S2") & (P["model"] == "SIZE")].iloc[0].auc
    ax.axvline(size_auc, color=FS.MODEL_COL["SIZE"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks(range(len(keys)), [k[2] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC (95% CI)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    # ---- 3: single flaws
    box(0.655, 0.18, 0.325, 0.67, "One flaw at a time inflates AUC")
    ax = fig.add_axes([0.80, 0.30, 0.165, 0.44])
    ax.set_zorder(2)
    ax.patch.set_alpha(0)
    pairs = [("Slice-level split (CNN)", ("CNN", "Strict (patient-grouped CV), slice level"),
              ("CNN", "L1 Slice-level random split")),
             ("Selection on all data", ("Radiomics", "Strict (patient-grouped CV)"),
              ("Radiomics", "L3 Feature selection on all data")),
             ("Best of many configs", ("Radiomics", "Strict, repeats 0-1"), ("Radiomics", "L5 Best of")),
             ("Epoch picked on test", ("CNN", "LOCO, epoch chosen on validation"),
              ("CNN", "L5 LOCO, epoch chosen on held-out centre"))]
    labels = []
    for i, (lab, ref, leak) in enumerate(pairs):
        def get(g, prefix):
            r = A[(A["group"] == g) & A["variant"].astype(str).str.startswith(prefix)]
            return np.nan if r.empty else float(r.iloc[0].auc)
        a0, a1 = get(*ref), get(*leak)
        ax.annotate("", xy=(a1, i), xytext=(a0, i),
                    arrowprops=dict(arrowstyle="-|>", color="#7F7F7F", lw=0.9, mutation_scale=6))
        ax.plot(a0, i, "o", ms=4.5, color=FS.STRICT_COL)
        ax.plot(a1, i, "o", ms=4.5, color=FS.LEAK_COL)
        ax.text(a1, i - 0.32, f"{a1:.2f}", ha="center", fontsize=6.5, color=FS.LEAK_COL)
        labels.append(lab)
    ax.set_yticks(range(len(pairs)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("AUC: strict (blue) → flawed (red)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    # ---- conclusion band
    fig.patches.append(FancyBboxPatch((0.02, 0.03), 0.96, 0.11, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, fc="#1F3A5F", ec="none", zorder=0))
    fig.text(0.5, 0.085, "Without evaluation leakage, T2WI radiomics and deep learning did not detect muscle invasion "
             "better than tumour size;\nsingle evaluation flaws inflated AUCs up to the near-perfect values reported "
             "in the literature.", ha="center", va="center", fontsize=7.5, color="white")
    out = C.FIGURES / "Graphical_abstract"
    FS.save(fig, out, align=False)
    print("graphical abstract written:", out)


if __name__ == "__main__":
    main()
