"""02 - Image preprocessing (label-free, per scan; no information crosses cases).

For each unique scan: N4 bias-field correction, then in-plane resampling to 0.7 mm
(slice thickness kept). For each tumour: resampled tumour mask, 3-mm in-plane
peritumoural ring, and two perturbed masks (1-voxel 2-D erosion / dilation) used
for the feature-robustness (ICC) filter.
Outputs in data/interim/{vol,mask,ring,mask_ero,mask_dil,ring_ero,ring_dil}/ and
data/processed/preprocess_log.csv.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from bcmri import config as C

OUT = {k: C.INTERIM / k for k in ("vol", "mask", "ring", "mask_ero", "mask_dil", "ring_ero", "ring_dil")}
for d in OUT.values():
    d.mkdir(parents=True, exist_ok=True)


def slice_axis_last(img):
    """Permute so the thickest axis is the third image axis (numpy axis 0)."""
    sp = np.array(img.GetSpacing())
    k = int(np.argmax(sp))
    if k == 2:
        return img, (0, 1, 2)
    order = [i for i in range(3) if i != k] + [k]
    return sitk.PermuteAxes(img, order), tuple(order)


def n4(img):
    img = sitk.Cast(img, sitk.sitkFloat32)
    shrink = [4, 4, 1]
    small = sitk.Shrink(img, shrink)
    fg = sitk.OtsuThreshold(small, 0, 1, 200)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([50, 40, 30])
    corr.Execute(small, fg)
    log_bias = corr.GetLogBiasFieldAsImage(img)
    return sitk.Cast(img / sitk.Exp(log_bias), sitk.sitkFloat32)


def resample_inplane(img, spacing_xy, interp, ref=None):
    sp = img.GetSpacing()
    new_sp = (spacing_xy, spacing_xy, sp[2])
    size = img.GetSize()
    new_size = [int(round(size[0] * sp[0] / new_sp[0])), int(round(size[1] * sp[1] / new_sp[1])), size[2]]
    if ref is not None:
        return sitk.Resample(img, ref, sitk.Transform(), interp, 0)
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(new_sp)
    f.SetSize(new_size)
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(interp)
    f.SetDefaultPixelValue(0)
    return f.Execute(img)


def ring_2d(mask_zyx, px, other=None):
    """In-plane ring of `px` pixels around the tumour, slice by slice (no dilation across slices)."""
    st = ndimage.generate_binary_structure(2, 1)
    st = ndimage.iterate_structure(st, 1)
    out = np.zeros_like(mask_zyx, dtype=bool)
    yy, xx = np.mgrid[-px:px + 1, -px:px + 1]
    disk = (yy ** 2 + xx ** 2) <= px ** 2
    for z in range(mask_zyx.shape[0]):
        m = mask_zyx[z]
        if m.any():
            out[z] = ndimage.binary_dilation(m, structure=disk) & ~m
    if other is not None:
        out &= ~other
    return out


def morph_2d(mask_zyx, op):
    st = ndimage.generate_binary_structure(2, 1)
    out = np.zeros_like(mask_zyx, dtype=bool)
    for z in range(mask_zyx.shape[0]):
        m = mask_zyx[z]
        if m.any():
            out[z] = ndimage.binary_erosion(m, st) if op == "ero" else ndimage.binary_dilation(m, st)
    if op == "ero" and out.sum() < 10:      # tiny tumour: keep the original rather than an empty mask
        out = mask_zyx.copy()
    return out


def to_img(arr, ref, dtype=sitk.sitkUInt8):
    im = sitk.GetImageFromArray(arr.astype(np.uint8))
    im.CopyInformation(ref)
    return sitk.Cast(im, dtype)


def read_scalar(path):
    """Read a T2WI volume as a scalar float32 image (some C3 files carry a 1-component vector type)."""
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(img)
    if a.ndim == 4:
        a = a[..., 0]
    out = sitk.GetImageFromArray(a.astype(np.float32))
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())
    return out


def main():
    df = pd.read_csv(C.PROCESSED / "cohort.csv")
    root = C.EXTRACTED
    logs = []
    ring_px = int(round(C.RING_MM / C.INPLANE_SPACING))
    only = set(sys.argv[1:])
    for scan_id, g in df.groupby("scan_id", sort=False):
        if only and scan_id not in only:
            continue
        t0 = time.time()
        vol_out = OUT["vol"] / f"{scan_id}.nii.gz"
        img0 = read_scalar(root / g.iloc[0]["image_path"])
        img0, perm = slice_axis_last(img0)
        if vol_out.exists():
            vol = sitk.ReadImage(str(vol_out))
        else:
            vol = resample_inplane(n4(img0), C.INPLANE_SPACING, sitk.sitkBSpline)
            vol = sitk.Cast(sitk.Clamp(vol, sitk.sitkFloat32, 0), sitk.sitkFloat32)   # B-spline undershoot
            sitk.WriteImage(vol, str(vol_out), useCompression=True)
        masks = {}
        for _, r in g.iterrows():
            m0 = sitk.ReadImage(str(root / r["mask_path"]))
            if perm != (0, 1, 2):
                m0 = sitk.PermuteAxes(m0, perm)
            m0 = sitk.Cast(m0 > 0, sitk.sitkUInt8)
            m0.CopyInformation(img0)        # geometry verified identical in 01_inventory
            m = resample_inplane(m0, C.INPLANE_SPACING, sitk.sitkNearestNeighbor, ref=vol)
            masks[r["case_id"]] = sitk.GetArrayFromImage(m) > 0
        union = np.zeros_like(next(iter(masks.values())))
        for a in masks.values():
            union |= a
        for cid, a in masks.items():
            others = union & ~a
            ero, dil = morph_2d(a, "ero"), morph_2d(a, "dil")
            arrs = {"mask": a, "ring": ring_2d(a, ring_px, others),
                    "mask_ero": ero, "mask_dil": dil,
                    "ring_ero": ring_2d(ero, ring_px, others), "ring_dil": ring_2d(dil, ring_px, others)}
            for k, arr in arrs.items():
                sitk.WriteImage(to_img(arr, vol), str(OUT[k] / f"{cid}.nii.gz"), useCompression=True)
            logs.append({"case_id": cid, "scan_id": scan_id, "perm": "".join(map(str, perm)),
                         "vol_size": "x".join(map(str, vol.GetSize())),
                         "spacing": ",".join(f"{s:.3f}" for s in vol.GetSpacing()),
                         "mask_vox": int(a.sum()), "ring_vox": int(arrs["ring"].sum()),
                         "ero_vox": int(ero.sum()), "dil_vox": int(dil.sum()), "sec": round(time.time() - t0, 1)})
        print(f"{scan_id}: {len(g)} tumour(s), {time.time() - t0:.1f}s", flush=True)
    out = pd.DataFrame(logs)
    f = C.PROCESSED / "preprocess_log.csv"
    if only and f.exists():
        old = pd.read_csv(f)
        out = pd.concat([old[~old["case_id"].isin(out["case_id"])], out])
    out.to_csv(f, index=False)


if __name__ == "__main__":
    main()
