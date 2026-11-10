#!/bin/bash
# Assemble the upload folder for Insights into Imaging (run from the project root: bash manuscript/make_submission.sh).
# Run first: code/12_figures.py, code/13_tables.py, code/14_values.py, manuscript/build.py, manuscript/build_supp.py,
# and fill in the date on the first line of manuscript/cover_letter.md.
#
# submission_InsightsImaging/
#   Manuscript.docx                 editable main text (title page, abstract, key points, text, references,
#                                   tables, figure legends; no embedded figures)
#   Manuscript_with_figures.pdf     the same with figures embedded (for checking; images 300 dpi, lossless)
#   Figures/Fig1-Fig6 *.tiff/.pdf   600-dpi TIFF (LZW) and vector PDF, 174 mm wide, Arial, editable text
#   Graphical_abstract.*            graphical abstract
#   Supplementary_Material.pdf/.docx, Supplementary_Tables.xlsx
#   Cover_letter.docx/.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
SIB=../BLCA_Rloop                                  # pandoc, typst and ghostscript installs shared with that project
PANDOC=${PANDOC:-$(command -v pandoc || echo "$SIB/conda_env/bin/pandoc")}
TYPST=${TYPST:-$(command -v typst || echo "$SIB/env/typst_env/bin/typst")}
GS=${GS:-$(command -v gs || echo "$SIB/conda_env/bin/gs")}
B=manuscript/build
P=submission_InsightsImaging
TMP=$(mktemp -d "${TMPDIR:-/tmp}/subm.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$P/Figures"

# Cover letter: {{KEY}} placeholders filled from values.tsv / values_results.tsv, then Word (editable) and PDF
# (A4, Times New Roman 11 pt, 2.2 cm margins, left-aligned).
python3 -I -c "
import sys; sys.path.insert(0, 'manuscript'); import mstools as m
w = []; t = m.fill_values(open('manuscript/cover_letter.md', encoding='utf-8').read(), m.read_values(), w)
open('$B/cover_letter.filled.md', 'w', encoding='utf-8').write(t); [print('WARNING (cover letter):', x) for x in w]"
"$PANDOC" "$B/cover_letter.filled.md" -o "$B/Cover_letter.docx" --reference-doc "$B/reference.docx"
printf 'margin:\n  x: 2.2cm\n  y: 2.2cm\n' > "$TMP/meta.yaml"
printf '#set par(justify: false)\n' > "$TMP/before.typ"
"$PANDOC" "$B/cover_letter.filled.md" -t typst -s -o "$B/cover_letter.typ" \
  -V papersize=a4 -V "mainfont=Times New Roman" -V fontsize=11pt --metadata-file="$TMP/meta.yaml" \
  -V 'header-includes=#set text(fallback: true)' --include-before-body="$TMP/before.typ"
"$TYPST" compile --root / "$B/cover_letter.typ" "$B/Cover_letter.pdf"

# Review PDF: images downsampled losslessly (Flate, no JPEG) from 600 to 300 dpi to keep the file small.
"$GS" -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.7 \
  -dDownsampleColorImages=true -dColorImageDownsampleType=/Bicubic -dColorImageResolution=300 \
  -dColorImageDownsampleThreshold=1.0 -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode \
  -dDownsampleGrayImages=true -dGrayImageDownsampleType=/Bicubic -dGrayImageResolution=300 \
  -dGrayImageDownsampleThreshold=1.0 -dAutoFilterGrayImages=false -dGrayImageFilter=/FlateEncode \
  -sOutputFile="$P/Manuscript_with_figures.pdf" "$B/Manuscript.pdf"

cp "$B/Manuscript.docx" "$B/Cover_letter.docx" "$B/Cover_letter.pdf" \
   "$B/Supplementary_Material.pdf" "$B/Supplementary_Material.docx" "$B/Supplementary_Tables.xlsx" "$P/"
for n in 1 2 3 4 5 6; do
  for ext in tiff pdf; do
    f=$(ls results/figures/Fig${n}_*.${ext} 2>/dev/null | head -1)
    if [ -n "$f" ]; then cp "$f" "$P/Figures/Fig${n}.${ext}"; else echo "WARNING: Fig${n}.${ext} missing"; fi
  done
done
for ext in tiff png pdf; do
  if [ -f "results/figures/Graphical_abstract.${ext}" ]; then cp "results/figures/Graphical_abstract.${ext}" "$P/"; fi
done
if grep -q '^\[Date\]' manuscript/cover_letter.md; then echo "WARNING: the cover letter date is not filled in"; fi
if grep -q 'To be confirmed' manuscript/values.tsv; then echo "WARNING: author details in manuscript/values.tsv are not confirmed"; fi
ls -lR "$P"
