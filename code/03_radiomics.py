"""03 - Handcrafted radiomics (PyRadiomics) for the tumour (IT) and the 3-mm ring (PT),
on the original mask and on two perturbed masks (1-voxel 2-D erosion / dilation) for ICC.

Outputs: data/processed/radiomics_{IT,PT}_{orig,ero,dil}.csv (index = case_id)
         data/processed/radiomics_failures.csv
"""
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from bcmri import config as C

PARAMS = {"IT": C.CODE / "params_radiomics_IT.yaml", "PT": C.CODE / "params_radiomics_PT.yaml"}
MASK_DIR = {("IT", "orig"): "mask", ("IT", "ero"): "mask_ero", ("IT", "dil"): "mask_dil",
            ("PT", "orig"): "ring", ("PT", "ero"): "ring_ero", ("PT", "dil"): "ring_dil"}
_EXT = {}


def _init():
    import radiomics
    from radiomics import featureextractor
    import SimpleITK as sitk
    radiomics.setVerbosity(logging.ERROR)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    for k, p in PARAMS.items():
        _EXT[k] = featureextractor.RadiomicsFeatureExtractor(str(p))


def _one(args):
    case_id, scan_id = args
    out, fails = {}, []
    vol = str(C.INTERIM / "vol" / f"{scan_id}.nii.gz")
    for (region, version), d in MASK_DIR.items():
        try:
            res = _EXT[region].execute(vol, str(C.INTERIM / d / f"{case_id}.nii.gz"), label=1)
            out[(region, version)] = {k: float(v) for k, v in res.items() if not k.startswith("diagnostics_")}
        except Exception as e:
            fails.append({"case_id": case_id, "region": region, "version": version, "error": repr(e)[:300]})
    return case_id, out, fails


def main():
    df = pd.read_csv(C.PROCESSED / "cohort.csv")
    jobs = list(zip(df["case_id"], df["scan_id"]))
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    tables = {k: {} for k in MASK_DIR}
    failures = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init) as pool:
        for i, (cid, out, fails) in enumerate(pool.imap_unordered(_one, jobs, chunksize=1)):
            for k, feats in out.items():
                tables[k][cid] = feats
            failures += fails
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(jobs)} cases, {time.time() - t0:.0f}s", flush=True)
    for (region, version), rows in tables.items():
        t = pd.DataFrame.from_dict(rows, orient="index")
        t.index.name = "case_id"
        t = t.reindex(df["case_id"])
        t.to_csv(C.PROCESSED / f"radiomics_{region}_{version}.csv")
        print(region, version, t.shape, "missing rows:", int(t.isna().all(axis=1).sum()), flush=True)
    pd.DataFrame(failures, columns=["case_id", "region", "version", "error"]).to_csv(
        C.PROCESSED / "radiomics_failures.csv", index=False)
    print(f"failures: {len(failures)}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
