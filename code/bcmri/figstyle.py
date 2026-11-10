"""Shared figure style: Arial 7 pt, editable text, Springer double-column width (174 mm)."""
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

MM = 1 / 25.4
FULL_W = 174 * MM          # Springer double column
HALF_W = 84 * MM           # Springer single column

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

# semantic colours (Okabe-Ito based; colour-blind safe)
MODEL_COL = {
    "RAD": "#0072B2",        # radiomics (primary)
    "CNN": "#D55E00",        # fine-tuned CNN (primary)
    "DLR": "#CC79A7",        # deep-radiomics fusion (primary)
    "SIZE": "#4D4D4D",       # tumour-size baseline
    "PROBE": "#009E73",      # frozen-feature linear probes
    "LATE": "#56B4E9",
    "CENTRE": "#9E9E9E",
}
CENTRE_COL = {"C1": "#0072B2", "C2": "#E69F00", "C3": "#009E73", "C4": "#CC79A7"}
LABEL_COL = {0: "#A6A6A6", 1: "#B2182B"}       # NMIBC, MIBC
LEAK_COL = "#B2182B"
STRICT_COL = "#0072B2"
GRID_GREY = "#E6E6E6"


def panel_label(ax, label, x=0, y=1, dx_pt=-14, dy_pt=4, **kw):
    """Bold lower-case panel label at a fixed physical offset from the axes corner."""
    off = ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + off, fontsize=8, fontweight="bold",
            ha="left", va="bottom", **kw)


def _alignment_gate():
    """Optional render-time panel-alignment auditor (QA only).

    If the auditor is available, set BCMRI_ALIGN_SCRIPTS to the directory that
    contains audit_panel_alignment.py, or place that module in env/figure_qa/.
    Analysis results do not depend on this check.
    """
    import os
    candidates = []
    env = os.environ.get("BCMRI_ALIGN_SCRIPTS")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[2] / "env" / "figure_qa")
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        return require_matplotlib_panel_alignment
    except Exception:  # noqa: BLE001
        return None


def save(fig, path, align=True, **align_kw):
    """Alignment gate (multi-panel figures), then SVG + PDF + 600-dpi TIFF + PNG preview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if align:
        require_matplotlib_panel_alignment = _alignment_gate()
        if require_matplotlib_panel_alignment is None:
            print(f"WARNING: alignment gate not available; {path.name} not audited")
        else:
            qa = path.parent / "qa"
            qa.mkdir(exist_ok=True)
            require_matplotlib_panel_alignment(fig, json_out=str(qa / f"{path.stem}.alignment.json"),
                                               overlay_svg=str(qa / f"{path.stem}.alignment.svg"),
                                               tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **align_kw)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
