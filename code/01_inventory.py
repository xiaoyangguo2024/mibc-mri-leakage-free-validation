"""01 - Data audit of the extracted FedBCa archive.

Layout found in the archive: Center{1-4}/T2WI/*.nii.gz (one file per scan),
Center{1-4}/Annotation/*.nii.gz (one mask per tumour), Center{1-4}/Center{c}_label.xlsx
with columns label, image_name, mask_name. Several masks can point to one image
(a patient with multiple tumours). Grouping unit = scan; scans that are hash-identical
or near-identical under another name are merged into one patient.

Outputs:
  data/processed/cohort.csv          one row per tumour
  data/processed/scan_stats.csv      one row per scan (geometry, intensity summary)
  results/tables/T0_data_audit.csv   per-centre summary vs the data paper
  results/logs/01_inventory.txt      audit log
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from bcmri import config as C

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def stem(name):
    return str(name).strip().replace(".nii.gz", "").replace(".nii", "")


def thumb(arr, size=48):
    z = arr.shape[0] // 2
    s = arr[max(0, z - 1):z + 2].astype(np.float32).mean(axis=0)
    t = ndimage.zoom(s, (size / s.shape[0], size / s.shape[1]), order=1)
    return ((t - t.mean()) / (t.std() + 1e-6)).ravel()


def orient_code(img):
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def main():
    root = C.EXTRACTED
    rows = []
    for c in range(1, 5):
        cdir = root / f"Center{c}"
        tab = pd.read_excel(cdir / f"Center{c}_label.xlsx")
        log(f"[C{c}] {len(tab)} rows; columns {list(tab.columns)}; labels {tab['label'].value_counts().to_dict()}")
        listed_img = set(tab["image_name"].astype(str).str.strip())
        listed_msk = set(tab["mask_name"].astype(str).str.strip())
        on_disk_img = {p.name for p in (cdir / "T2WI").glob("*.nii*")}
        on_disk_msk = {p.name for p in (cdir / "Annotation").glob("*.nii*")}
        log(f"[C{c}] images on disk {len(on_disk_img)}, listed {len(listed_img)}, unlisted {sorted(on_disk_img - listed_img)}, "
            f"missing {sorted(listed_img - on_disk_img)}")
        log(f"[C{c}] masks on disk {len(on_disk_msk)}, listed {len(listed_msk)}, unlisted {sorted(on_disk_msk - listed_msk)}, "
            f"missing {sorted(listed_msk - on_disk_msk)}")
        for _, r in tab.iterrows():
            rows.append({"centre": f"C{c}", "label": int(r["label"]),
                         "image_name": str(r["image_name"]).strip(), "mask_name": str(r["mask_name"]).strip(),
                         "scan_id": f"C{c}_{stem(r['image_name'])}", "case_id": f"C{c}_{stem(r['mask_name'])}",
                         "image_path": f"Center{c}/T2WI/{str(r['image_name']).strip()}",
                         "mask_path": f"Center{c}/Annotation/{str(r['mask_name']).strip()}"})
    df = pd.DataFrame(rows)
    assert df["case_id"].is_unique, "duplicate case ids"

    # ---- per scan
    scans = []
    thumbs = {}
    for sid, g in df.groupby("scan_id", sort=False):
        img = sitk.ReadImage(str(root / g.iloc[0]["image_path"]))
        a = sitk.GetArrayFromImage(img)
        fg = a > np.percentile(a, 50)
        scans.append({"scan_id": sid, "centre": g.iloc[0]["centre"], "n_tumours": len(g),
                      "size": "x".join(map(str, img.GetSize())),
                      "sx": img.GetSpacing()[0], "sy": img.GetSpacing()[1], "sz": img.GetSpacing()[2],
                      "orient": orient_code(img), "pixel_type": img.GetPixelIDTypeAsString(),
                      "origin": ",".join(f"{v:.2f}" for v in img.GetOrigin()),
                      "img_md5": hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest(),
                      "int_min": float(a.min()), "int_p50": float(np.percentile(a, 50)),
                      "int_p99": float(np.percentile(a, 99)), "int_max": float(a.max()),
                      "fg_mean": float(a[fg].mean()), "fg_sd": float(a[fg].std())})
        thumbs[sid] = (img.GetSize(), thumb(a))
    sc = pd.DataFrame(scans)

    # ---- duplicate / near-duplicate scans under different names
    parent = {s: s for s in sc["scan_id"]}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for h, g in sc.groupby("img_md5"):
        if len(g) > 1:
            log("hash-identical scans:", list(g["scan_id"]))
            for s in g["scan_id"].iloc[1:]:
                parent[find(s)] = find(g["scan_id"].iloc[0])
    ids = list(thumbs)
    best = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (si, ti), (sj, tj) = thumbs[ids[i]], thumbs[ids[j]]
            r = float(np.corrcoef(ti, tj)[0, 1])
            best.append((r, ids[i], ids[j], si == sj))
            if r > 0.98 and si == sj:
                log(f"near-identical scans (r={r:.4f}, same size): {ids[i]} ~ {ids[j]}")
                parent[find(ids[j])] = find(ids[i])
    best.sort(reverse=True)
    log("top-5 thumbnail correlations between different scans:", [(round(r, 3), a, b, same) for r, a, b, same in best[:5]])
    sc["patient_id"] = [find(s) for s in sc["scan_id"]]
    df = df.merge(sc[["scan_id", "patient_id"]], on="scan_id", how="left")

    # ---- per tumour
    recs = []
    for k, r in df.iterrows():
        img = sitk.ReadImage(str(root / r["image_path"]))
        msk = sitk.ReadImage(str(root / r["mask_path"]))
        m = sitk.GetArrayFromImage(msk)
        mb = m > 0
        sp = img.GetSpacing()
        lab, ncomp = ndimage.label(mb)
        zs = np.where(mb.any(axis=(1, 2)))[0]
        recs.append({
            "mask_size_match": msk.GetSize() == img.GetSize(),
            "mask_spacing_match": bool(np.allclose(msk.GetSpacing(), sp, atol=1e-3)),
            "mask_origin_diff_mm": float(np.max(np.abs(np.array(msk.GetOrigin()) - np.array(img.GetOrigin())))),
            "mask_dir_match": bool(np.allclose(msk.GetDirection(), img.GetDirection(), atol=1e-3)),
            "mask_values": ",".join(str(int(v)) for v in np.unique(m)[:6]),
            "mask_voxels": int(mb.sum()), "tumour_ml": float(mb.sum() * np.prod(sp) / 1000),
            "mask_components": int(ncomp), "n_slices_tumour": int(len(zs)),
            "z_first": int(zs.min()) if len(zs) else -1, "z_last": int(zs.max()) if len(zs) else -1,
            "z_gaps": int((np.diff(zs) > 1).sum()) if len(zs) > 1 else 0,
            "max_slice_area_mm2": float(mb.sum(axis=(1, 2)).max() * sp[0] * sp[1]) if len(zs) else 0.0,
            "mask_md5": hashlib.md5(np.ascontiguousarray(mb).tobytes()).hexdigest()})
    df = pd.concat([df, pd.DataFrame(recs, index=df.index)], axis=1)
    dup = df[df.duplicated(["scan_id", "mask_md5"], keep=False)]
    log(f"tumours with an identical mask on the same scan (true duplicates): {len(dup)}", list(dup["case_id"]))
    # overlapping masks within a scan
    for sid, g in df.groupby("scan_id"):
        if len(g) > 1:
            arrs = [sitk.GetArrayFromImage(sitk.ReadImage(str(root / p))) > 0 for p in g["mask_path"]]
            tot = np.sum(arrs, axis=0)
            if (tot > 1).any():
                log(f"overlapping masks in {sid}: {int((tot > 1).sum())} voxels")
    df["tumours_in_patient"] = df.groupby("patient_id")["case_id"].transform("size")
    df["mibc_in_patient"] = df.groupby("patient_id")["label"].transform("sum")
    df.to_csv(C.PROCESSED / "cohort.csv", index=False)
    sc.to_csv(C.PROCESSED / "scan_stats.csv", index=False)

    # ---- concordance of labels within multi-tumour patients
    mt = df[df["tumours_in_patient"] > 1].groupby("patient_id")["label"].agg(["size", "sum"])
    log(f"multi-tumour patients: {len(mt)}; all-NMIBC {int((mt['sum'] == 0).sum())}, all-MIBC "
        f"{int((mt['sum'] == mt['size']).sum())}, mixed {int(((mt['sum'] > 0) & (mt['sum'] < mt['size'])).sum())}")

    paper = {"C1": (160, 30, 119), "C2": (48, 25, 47), "C3": (32, 13, 24), "C4": (35, 17, 31)}
    out = []
    for code, g in df.groupby("centre"):
        s = sc[sc["centre"] == code]
        out.append({"centre": code, "scanner": C.SCANNER[code],
                    "tumours": len(g), "tumours_paper": paper[code][0],
                    "MIBC": int(g["label"].sum()), "MIBC_paper": paper[code][1],
                    "MIBC_pct": round(100 * g["label"].mean(), 1),
                    "scans": len(s), "patients": g["patient_id"].nunique(), "patients_paper_table1": paper[code][2],
                    "multi_tumour_patients": int((g.groupby("patient_id").size() > 1).sum()),
                    "max_tumours_per_patient": int(g.groupby("patient_id").size().max()),
                    "inplane_mm_median": round(float(np.median(s["sx"])), 3),
                    "inplane_mm_range": f"{s['sx'].min():.3f}-{s['sx'].max():.3f}",
                    "slice_mm_median": round(float(np.median(s["sz"])), 2),
                    "slice_mm_range": f"{s['sz'].min():.2f}-{s['sz'].max():.2f}",
                    "matrix_top": s["size"].value_counts().index[0],
                    "orientation": "/".join(f"{k}:{v}" for k, v in s["orient"].value_counts().items()),
                    "pixel_type": "/".join(s["pixel_type"].unique()),
                    "tumour_ml_median": round(float(np.median(g["tumour_ml"])), 2),
                    "tumour_ml_IQR": f"{np.percentile(g['tumour_ml'], 25):.2f}-{np.percentile(g['tumour_ml'], 75):.2f}",
                    "slices_median": float(np.median(g["n_slices_tumour"])),
                    "mask_geometry_mismatch": int((~(g["mask_size_match"] & g["mask_spacing_match"] & g["mask_dir_match"])
                                                   | (g["mask_origin_diff_mm"] > 0.01)).sum()),
                    "multi_component_masks": int((g["mask_components"] > 1).sum()),
                    "empty_masks": int((g["mask_voxels"] == 0).sum())})
    out = pd.DataFrame(out)
    out.to_csv(C.TABLES / "T0_data_audit.csv", index=False)
    log(out.T.to_string())
    log(f"TOTAL tumours {len(df)}, MIBC {int(df['label'].sum())}, scans {len(sc)}, patients {df['patient_id'].nunique()}")
    (C.RESULTS / "logs").mkdir(exist_ok=True)
    (C.RESULTS / "logs" / "01_inventory.txt").write_text("\n".join(LOG))


if __name__ == "__main__":
    main()
