#!/usr/bin/env bash
# Build the code-and-derived-data archive for deposit (e.g. Zenodo / GitHub release). Nothing is uploaded.
# Usage: bash make_code_archive.sh      -> dist/BLCA_MRI_MIBC_code_<date>.zip
# Excluded: raw and preprocessed images (data/raw, data/interim; obtain FedBCa from Zenodo 10409145),
# the Python environment (env/, except the pinned package lists and the weight manifest), CNN weight files
# (results/models/*.pt, 45 MB each; written again by code/07_cnn.py), tumour-centred MRI crops
# (data/processed/gradcam_examples.npz; written again by code/10_interpret.py), manuscript build outputs and local working files.
set -euo pipefail
cd "$(dirname "$0")"
NAME="BLCA_MRI_MIBC_code_$(date +%Y%m%d)"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/codearch.XXXXXX")/${NAME}"
mkdir -p "$STAGE" dist

X=(--exclude .DS_Store --exclude __pycache__ --exclude "*.pyc")  # OS and Python caches
copy() { for p in "$@"; do [ -e "$p" ] && rsync -a --relative "${X[@]}" "$p" "$STAGE/"; done; }

copy README.md LICENSE .zenodo.json ANALYSIS_PLAN_en.md 分析方案.md run_step.sh make_code_archive.sh
copy env/environment_explicit.txt env/requirements_pip.txt env/weights/MANIFEST.tsv
copy data/external
rsync -a --relative "${X[@]}" --exclude gradcam_examples.npz data/processed "$STAGE/"
copy results/tables results/figures results/predictions results/logs results/models/final_rad_itpt_lr.pkl
copy logs
rsync -a "${X[@]}" code "$STAGE/"
rsync -a "${X[@]}" --exclude build manuscript "$STAGE/"

# logs: replace this machine's project path with "."
find "$STAGE" -type f \( -name '*.log' -o -name '*.txt' \) -print0 | xargs -0 perl -pi -e "s#\Q$PWD\E#.#g"

# sanity checks: no raw images, no environment, no oversized files, no local home path
if find "$STAGE" \( -name '*.nii*' -o -name '*.nrrd' -o -name '*.dcm' -o -name '*.pt' \) | grep -q .; then
  echo "ERROR: image or weight files found in the archive stage" >&2; exit 1
fi
big=$(find "$STAGE" -type f -size +50M)
if [ -n "$big" ]; then echo "ERROR: files > 50 MB:" >&2; echo "$big" >&2; exit 1; fi
if grep -rIl "$HOME" "$STAGE" | grep -q .; then echo "WARNING: local home path still in:" >&2; grep -rIl "$HOME" "$STAGE" >&2; fi

( cd "$(dirname "$STAGE")" && zip -qr -X "$OLDPWD/dist/${NAME}.zip" "$NAME" )
rm -rf "$(dirname "$STAGE")"
echo "wrote dist/${NAME}.zip ($(du -h "dist/${NAME}.zip" | cut -f1))"
unzip -l "dist/${NAME}.zip" | tail -1
