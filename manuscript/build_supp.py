"""Build the Supplementary Material (methods, checklists, small tables, figures, references).

    python3 -I manuscript/build_supp.py           # run from the project root, after code/13_tables.py and build.py

Reads manuscript/supplementary.md, fills {{KEY}} placeholders (values.tsv, values_results.tsv), numbers its own
citations (a separate reference list at the end), inserts <!-- TABLE: ... --> and <!-- FIGURE: ... --> markers and
writes manuscript/build/Supplementary_Material.docx and .pdf. Copies results/tables/Supplementary_Tables.xlsx
(every supplementary table, one sheet each) to manuscript/build/.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mstools as mst

os.makedirs(mst.OUT, exist_ok=True)
warnings = []

text = open(os.path.join(mst.MS, "supplementary.md"), encoding="utf-8").read()
text = mst.fill_values(text, mst.read_values(), warnings)
keymap, cache = mst.load_refs()
cite = mst.Citer(keymap, cache)
text = mst.CITE.sub(cite, text)
if cite.unknown:
    warnings.append("citation keys missing: " + ", ".join(sorted(cite.unknown)))
text = text.replace("<!-- REFERENCES -->", cite.reference_list())
text = mst.replace_markers(text, warnings)
md_path = os.path.join(mst.OUT, "supplementary.md")
open(md_path, "w", encoding="utf-8").write(text)

ref_docx = os.path.join(mst.OUT, "reference.docx")      # made by build.py (Times New Roman styles)
docx = os.path.join(mst.OUT, "Supplementary_Material.docx")
cmd = [mst.PANDOC, md_path, "-f", "markdown-citations", "-o", docx, "--resource-path", mst.ROOT]
if os.path.exists(ref_docx):
    cmd += ["--reference-doc", ref_docx]
else:
    warnings.append("reference.docx missing: run build.py first for Times New Roman styles")
subprocess.run(cmd, check=True)

pdf = None
if mst.TYPST:
    typ = os.path.join(mst.OUT, "supplementary.typ")
    subprocess.run([mst.PANDOC, md_path, "-f", "markdown-citations", "-t", "typst", "-s", "-o", typ,
                    "-V", "papersize=a4", "-V", "mainfont=Times New Roman", "-V", "fontsize=10pt",
                    "-V", "margin.x=2cm", "-V", "margin.y=2cm", "-V", "page-numbering=1",
                    "-V", "header-includes=#set text(fallback: true)\n#show table: set text(size: 7.5pt)",
                    "--resource-path", mst.ROOT], check=True)
    pdf = os.path.join(mst.OUT, "Supplementary_Material.pdf")
    r = subprocess.run([mst.TYPST, "compile", "--root", "/", typ, pdf], capture_output=True, text=True)
    if r.returncode:
        warnings.append("typst failed: " + r.stderr[-800:])
        pdf = None
else:
    warnings.append("typst not found: PDF not built")

xlsx = os.path.join(mst.ROOT, "results", "tables", "Supplementary_Tables.xlsx")
if os.path.exists(xlsx):
    shutil.copy(xlsx, os.path.join(mst.OUT, "Supplementary_Tables.xlsx"))
else:
    warnings.append("table workbook missing (run code/13_tables.py supplementary): " + xlsx)

print(f"supplementary references: {len(cite.order)}")
print(f"wrote {docx}" + (f" and {pdf}" if pdf else ""))
if pdf:
    print(f"PDF size: {os.path.getsize(pdf) / 1e6:.1f} MB")
for w in warnings:
    print("WARNING:", w)
