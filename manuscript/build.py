"""Assemble the manuscript (Springer Nature numbered style, as used by Insights into Imaging / European Radiology).

    python3 -I manuscript/build.py                 # run from the project root
    PANDOC=/path/to/pandoc TYPST=/path/to/typst python3 -I manuscript/build.py

Steps
  1. concatenate manuscript/sections/*.md in file-name order
  2. fill {{KEY}} placeholders from manuscript/values.tsv (key<TAB>value)
  3. number [@key] citations by order of first appearance; compress runs ([1-3], [1,2])
  4. format the reference list (Springer basic style) from refs_cache.json (PubMed, made by fetch_refs.py)
     and refs_manual.json (references without a PMID, e.g. arXiv preprints and conference papers)
  5. replace <!-- TABLE: file.csv | transpose --> and <!-- FIGURE: file.png --> markers
  6. pandoc -> build/Manuscript.docx (Times New Roman 12 pt, 1.5 spacing, line and page numbers)
     pandoc -> typst -> build/Manuscript.pdf (figures embedded; the .docx has legends only)
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # -I does not add the script directory
import mstools as mst

ROOT, MS, OUT, PANDOC, TYPST = mst.ROOT, mst.MS, mst.OUT, mst.PANDOC, mst.TYPST
os.makedirs(OUT, exist_ok=True)
warnings = []

# ---------------------------------------------------------------- 1-2. text
sections = sorted(f for f in os.listdir(os.path.join(MS, "sections")) if f.endswith(".md"))
text = "\n\n".join(open(os.path.join(MS, "sections", f), encoding="utf-8").read().strip() for f in sections) + "\n"

values = mst.read_values()
text = mst.fill_values(text, values, warnings)

# ---------------------------------------------------------------- 3. citations
keymap, cache = mst.load_refs()
cite = mst.Citer(keymap, cache)
CITE = mst.CITE
order, unknown, number = cite.order, cite.unknown, cite


body, _, tail = text.partition("<!-- REFERENCES -->")
body = CITE.sub(number, body)
tail = CITE.sub(number, tail)          # tables and figure legends come after the reference list
if unknown:
    warnings.append("citation keys missing from references.tsv/refs_cache.json: " + ", ".join(sorted(unknown)))
stray = re.findall(r"\[@[^\]]*\]", body + tail)
if stray:
    warnings.append("unparsed citations: " + ", ".join(stray[:5]))
unused = sorted(set(keymap) - set(order))


# ---------------------------------------------------------------- 4. reference list
refs = sorted(order, key=order.get)
ref_md = cite.reference_list()
text = body + ref_md + tail


# ---------------------------------------------------------------- 5. tables and figures
# Word file for submission: figures are uploaded separately, so only their legends stay in the text.
# PDF for reading and review: figures embedded after the legends.
md_path = os.path.join(OUT, "manuscript.md")
open(md_path, "w", encoding="utf-8").write(mst.replace_markers(text, warnings, figures=False))
md_fig = os.path.join(OUT, "manuscript_with_figures.md")
_wf = []
open(md_fig, "w", encoding="utf-8").write(mst.replace_markers(text, _wf, figures=True))
warnings += [w for w in _wf if w.startswith("figure")]         # table warnings came from the first pass


# ---------------------------------------------------------------- 6. docx
def make_reference_docx(path):
    tmp = path + ".default.docx"
    subprocess.run([PANDOC, "-o", tmp, "--print-default-data-file", "reference.docx"], check=True)
    TNR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman" />'

    def style(s, sid, size, bold=True, italic=False):
        m = re.search(r'<w:style [^>]*w:styleId="%s">.*?</w:style>' % sid, s, re.S)
        if not m:
            return s
        b = m.group(0)
        b = re.sub(r'<w:sz w:val="\d+" ?/>', f'<w:sz w:val="{size}" />', b)
        b = re.sub(r'<w:szCs w:val="\d+" ?/>', f'<w:szCs w:val="{size}" />', b)
        b = re.sub(r"<w:[bi]\s*/>", "", b)
        b = b.replace("<w:rPr>", "<w:rPr>" + ("<w:b />" if bold else "") + ("<w:i />" if italic else ""), 1)
        return s[:m.start()] + b + s[m.end():]

    footer = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
              '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
              '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
              '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
              '<w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
    zin = zipfile.ZipFile(tmp)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            d = zin.read(item.filename).decode("utf-8")
            if item.filename == "word/styles.xml":
                d = re.sub(r"<w:rFonts\b[^>]*/>", TNR, d)
                d = re.sub(r"<w:color\b[^>]*/>", "", d)
                d = d.replace('<w:spacing w:after="200" />', '<w:spacing w:after="120" w:line="360" w:lineRule="auto" />', 1)
                d = style(d, "Title", 32)
                d = style(d, "Heading1", 28)
                d = style(d, "Heading2", 24)
                d = style(d, "Heading3", 24, bold=True, italic=True)
                d = style(d, "Heading4", 24, bold=False, italic=True)
            elif item.filename == "word/document.xml":
                d = re.sub(r"<w:sectPr>.*?</w:sectPr>",
                           '<w:sectPr><w:footerReference w:type="default" r:id="rIdFtr1" />'
                           '<w:footnotePr><w:numRestart w:val="eachSect" /></w:footnotePr>'
                           '<w:pgSz w:w="11906" w:h="16838" /><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" '
                           'w:left="1440" w:header="708" w:footer="708" w:gutter="0" />'
                           '<w:lnNumType w:countBy="1" w:restart="continuous" /></w:sectPr>', d, flags=re.S)
                if "xmlns:r=" not in d[:600]:
                    d = d.replace("<w:document ", '<w:document xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ', 1)
            elif item.filename == "word/_rels/document.xml.rels":
                d = d.replace("</Relationships>", '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                              'relationships/footer" Id="rIdFtr1" Target="footer1.xml" /></Relationships>')
            elif item.filename == "[Content_Types].xml":
                d = d.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/'
                              'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml" /></Types>')
            zout.writestr(item, d.encode("utf-8"))
        zout.writestr("word/footer1.xml", footer.encode("utf-8"))
    os.remove(tmp)


ref_docx = os.path.join(OUT, "reference.docx")
make_reference_docx(ref_docx)
docx = os.path.join(OUT, "Manuscript.docx")
subprocess.run([PANDOC, md_path, "-f", "markdown-citations", "-o", docx, "--reference-doc", ref_docx,
                "--resource-path", ROOT], check=True)

# ---------------------------------------------------------------- 6b. pdf
pdf = None
if TYPST:
    typ = os.path.join(OUT, "manuscript.typ")
    subprocess.run([PANDOC, md_fig, "-f", "markdown-citations", "-t", "typst", "-s", "-o", typ,
                    "-V", "papersize=a4", "-V", "mainfont=Times New Roman", "-V", "fontsize=12pt",
                    "-V", "linestretch=1.5", "-V", "page-numbering=1", "-V", "margin.x=2.54cm", "-V", "margin.y=2.54cm",
                    "-V", 'header-includes=#set par.line(numbering: "1", number-clearance: 1.2em)\n#set text(fallback: true)',
                    "--resource-path", ROOT], check=True)
    t = open(typ, encoding="utf-8").read()     # title block: no justification (avoids stretched word spacing)
    open(typ, "w", encoding="utf-8").write(t.replace("align(center, block[", "align(center, block[#set par(justify: false)\n", 1))
    pdf = os.path.join(OUT, "Manuscript.pdf")
    r = subprocess.run([TYPST, "compile", "--root", "/", typ, pdf], capture_output=True, text=True)
    if r.returncode:
        warnings.append("typst failed: " + r.stderr[-800:])
        pdf = None
else:
    warnings.append("typst not found: PDF not built (set TYPST=/path/to/typst, or save the .docx as PDF in Word)")

# ---------------------------------------------------------------- report
main = re.split(r"^# (?:Declarations|References)\b", body.split("# Introduction")[-1], flags=re.M)[0]
main = re.sub(r"^#.*$|\[[0-9,\-]+\]", " ", main, flags=re.M)        # headings and citation numbers do not count
words = len(re.findall(r"[A-Za-z0-9]+(?:['’.-][A-Za-z0-9]+)*", main)) if "# Introduction" in body else 0
print(f"sections: {', '.join(sections)}")
print(f"references cited: {len(refs)}; in references.tsv but not cited: {', '.join(unused) or 'none'}")
print(f"approx. main-text words (Introduction to end of Discussion; limit 3000): {words}")
print(f"wrote {docx}" + (f" and {pdf}" if pdf else ""))
for w in warnings:
    print("WARNING:", w)
