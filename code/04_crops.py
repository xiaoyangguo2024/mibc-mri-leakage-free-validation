"""04 - 2.5-D tumour-centred crops for the deep-learning models (label-free).

Per tumour: every slice that contains tumour, three channels = slices z-1, z, z+1, a square
field of view centred on the tumour's in-plane bounding box (side = max(80 mm, box + 2x10 mm)),
resized to 224 x 224. Intensities: per-scan robust scaling (0.5th-99.5th percentile of the
scan's foreground) to [0, 255]. Tumour and ring masks are cropped the same way (Grad-CAM).

Outputs: data/interim/crops/{case_id}.npz and data/processed/crops_index.csv (one row per slice).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from bcmri import config as C

OUT = C.INTERIM / "crops"
OUT.mkdir(parents=True, exist_ok=True)
FOV_MIN_MM = 80.0


def robust_scale(vol):
    fg = vol[vol > np.percentile(vol, 40)]           # drop air/background
    lo, hi = np.percentile(fg, [0.5, 99.5])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)


def crop2d(a, cy, cx, half, fill=0.0):
    """Square crop [cy-half, cy+half) x [cx-half, cx+half) with padding outside the image."""
    y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
    out = np.full((2 * half, 2 * half), fill, dtype=a.dtype)
    sy0, sy1, sx0, sx1 = max(0, y0), min(a.shape[0], y1), max(0, x0), min(a.shape[1], x1)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def resize(a, size, order):
    z = size / a.shape[0]
    return ndimage.zoom(a, z, order=order, prefilter=(order > 1))


def main():
    df = pd.read_csv(C.PROCESSED / "cohort.csv")
    rows = []
    for _, r in df.iterrows():
        cid, sid = r["case_id"], r["scan_id"]
        vol_img = sitk.ReadImage(str(C.INTERIM / "vol" / f"{sid}.nii.gz"))
        vol = sitk.GetArrayFromImage(vol_img).astype(np.float32)        # (z, y, x)
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(C.INTERIM / "mask" / f"{cid}.nii.gz"))) > 0
        ring = sitk.GetArrayFromImage(sitk.ReadImage(str(C.INTERIM / "ring" / f"{cid}.nii.gz"))) > 0
        sp = vol_img.GetSpacing()                                         # (x, y, z)
        v = robust_scale(vol)
        zs = np.where(msk.any(axis=(1, 2)))[0]
        ys, xs = np.where(msk.any(axis=0))
        cy, cx = int(round((ys.min() + ys.max()) / 2)), int(round((xs.min() + xs.max()) / 2))
        box_mm = max((ys.max() - ys.min() + 1) * sp[1], (xs.max() - xs.min() + 1) * sp[0])
        fov_mm = max(FOV_MIN_MM, box_mm + 2 * C.CROP_MARGIN_MM)
        half = int(round(fov_mm / sp[0] / 2))
        imgs, ms, rs, areas = [], [], [], []
        for z in zs:
            chans = []
            for dz in (-1, 0, 1):
                zz = int(np.clip(z + dz, 0, vol.shape[0] - 1))
                chans.append(resize(crop2d(v[zz], cy, cx, half), C.CROP_SIZE, 1))
            imgs.append(np.stack(chans))
            ms.append(resize(crop2d(msk[z].astype(np.float32), cy, cx, half), C.CROP_SIZE, 0) > 0.5)
            rs.append(resize(crop2d(ring[z].astype(np.float32), cy, cx, half), C.CROP_SIZE, 0) > 0.5)
            areas.append(float(msk[z].sum() * sp[0] * sp[1]))
        imgs = (np.clip(np.stack(imgs), 0, 1) * 255).round().astype(np.uint8)
        areas = np.asarray(areas)
        rank = np.argsort(np.argsort(-areas))                            # 0 = largest slice
        np.savez_compressed(OUT / f"{cid}.npz", img=imgs, mask=np.stack(ms).astype(np.uint8),
                            ring=np.stack(rs).astype(np.uint8), z=zs, area_mm2=areas, rank=rank,
                            mm_per_px=fov_mm / C.CROP_SIZE)
        for i, z in enumerate(zs):
            rows.append({"case_id": cid, "slice_idx": i, "z": int(z), "area_mm2": areas[i], "area_rank": int(rank[i]),
                         "area_frac_of_max": areas[i] / areas.max(), "fov_mm": fov_mm,
                         "slice_mm": sp[2]})
    idx = pd.DataFrame(rows)
    idx.to_csv(C.PROCESSED / "crops_index.csv", index=False)
    print(idx.groupby("case_id").size().describe())
    print("FOV > 80 mm:", int((idx.groupby("case_id")["fov_mm"].first() > FOV_MIN_MM + 1e-6).sum()), "tumours")


if __name__ == "__main__":
    main()
