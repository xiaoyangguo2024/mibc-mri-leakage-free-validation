"""Helpers shared by build.py (manuscript) and build_supp.py (supplementary material).

Values: {{KEY}} placeholders are filled from values.tsv (static) and values_results.tsv (written by
code/14_values.py). Citations: [@key; @key2] are numbered by first appearance and formatted in the Springer
basic style from refs_cache.json (PubMed records made by fetch_refs.py) and refs_manual.json.
"""
import csv
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "manuscript")
OUT = os.path.join(MS, "build")
_SIB = os.path.join(os.path.dirname(ROOT), "BLCA_Rloop")      # tool installs shared with a sibling project
PANDOC = os.environ.get("PANDOC") or shutil.which("pandoc") or os.path.join(_SIB, "conda_env", "bin", "pandoc")
TYPST = (os.environ.get("TYPST") or shutil.which("typst")
         or next((p for p in [os.path.join(_SIB, "env", "typst_env", "bin", "typst")] if os.path.exists(p)), None))
PAGEBREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n```{=typst}\n#pagebreak()\n```\n'
CITE = re.compile(r"\[(@[A-Za-z0-9_:-]+(?:\s*;\s*@[A-Za-z0-9_:-]+)*)\]")


# ---------------------------------------------------------------- values
def read_values():
    values = {}
    for vf in (os.path.join(MS, "values.tsv"), os.path.join(MS, "values_results.tsv")):   # static, then computed
        if os.path.exists(vf):
            for line in open(vf, encoding="utf-8"):
                k, _, v = line.rstrip("\n").partition("\t")
                if k and not k.startswith("#"):
                    values[k] = v.replace("\\n", "\n")      # a literal \n in the file is a line break
    return values


def fill_values(text, values, warnings):
    text = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)), text)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)))
    if left:
        warnings.append("unfilled placeholders: " + ", ".join(left))
    return text


# ---------------------------------------------------------------- citations
def load_refs():
    keymap = {}
    for line in open(os.path.join(MS, "references.tsv"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1].strip() and not parts[0].startswith("#"):
            keymap[parts[0].strip()] = parts[1].strip()      # a PMID, or the key of an entry in refs_manual.json
    cache = json.load(open(os.path.join(MS, "refs_cache.json"), encoding="utf-8"))
    man = os.path.join(MS, "refs_manual.json")
    if os.path.exists(man):
        cache.update(json.load(open(man, encoding="utf-8")))
    return keymap, cache


def compress(ns):
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}-{ns[j]}" if j - i >= 2 else ",".join(str(n) for n in ns[i:j + 1]))
        i = j + 1
    return ",".join(out)


class Citer:
    """re.sub callback that numbers [@key] citations by order of first appearance."""

    def __init__(self, keymap, cache):
        self.keymap, self.cache, self.order, self.unknown = keymap, cache, {}, set()

    def __call__(self, m):
        nums = []
        for k in (x.strip().lstrip("@") for x in m.group(1).split(";")):
            if k not in self.keymap or self.keymap[k] not in self.cache:
                self.unknown.add(k)
                continue
            self.order.setdefault(k, len(self.order) + 1)
            nums.append(self.order[k])
        return "[" + compress(sorted(set(nums))) + "]"

    def reference_list(self):
        refs = sorted(self.order, key=self.order.get)
        return "\n\n".join(f"{self.order[k]}\\. {md_escape(fmt_ref(self.cache[self.keymap[k]]))}" for k in refs)


def md_escape(s):
    return re.sub(r"([\\*_\[\]<>$@#`|~^])", r"\\\1", s)


def fmt_ref(r):
    """Springer basic style: Surname AB, Surname CD, Surname EF et al (Year) Title. Journal Vol:Pages. DOI"""
    au = r.get("authors") or []
    if not au:
        a = r.get("group") or "Anonymous"
    elif len(au) > 3:
        a = ", ".join(au[:3]) + " et al"
    else:
        a = ", ".join(au)
    t = re.sub(r"^\[(.*)\]\.?$", r"\1", r["title"].strip())
    t = t if t[-1:] in ".?!" else t + "."
    kind = r.get("type", "journal")
    doi = f" https://doi.org/{r['doi']}" if r.get("doi") else ""
    if kind == "journal":
        j = r["journal"].replace(".", "")
        pg = (r.get("pages") or r.get("elocation") or "").replace("-", "–")
        if r.get("volume"):
            src = f"{j} {r['volume']}:{pg}" if pg else f"{j} {r['volume']}"
        else:
            src = f"{j}"
        return f"{a} ({r['year']}) {t} {src}.{doi}"
    if kind == "conference":
        pg = f", pp {r['pages'].replace('-', chr(8211))}" if r.get("pages") else ""
        return f"{a} ({r['year']}) {t} In: {r['journal']}{pg}.{doi}"
    if kind == "preprint":
        return f"{a} ({r['year']}) {t} Preprint at {r['url']}"
    if kind == "software":
        return f"{a} ({r['year']}) {t} {r.get('journal', '')} {r['url']}".replace("  ", " ")
    raise ValueError(kind)


# ---------------------------------------------------------------- tables and figures
def csv_table(path, transpose=False):
    """Pipe table from a CSV (or TSV, by extension) file given relative to the project root."""
    full = os.path.join(ROOT, path)
    rows = list(csv.reader(open(full, encoding="utf-8"), delimiter="\t" if path.endswith(".tsv") else ","))
    if transpose:
        rows = [list(x) for x in zip(*rows)]
    cell = lambda s: s.replace("|", "\\|").strip() or "-"
    head, rest = rows[0], rows[1:]
    lines = ["| " + " | ".join(cell(h) for h in head) + " |",
             "|" + "|".join(":---" if i == 0 else ":---:" for i in range(len(head))) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rest]
    return "\n".join(lines)


def replace_markers(text, warnings, figures=True):
    """<!-- TABLE: file.csv | transpose -->, <!-- FIGURE: file.png --> and <!-- PAGEBREAK --> markers.

    figures=False drops the figure images (journals that want figures uploaded as separate files)."""
    def table_marker(m):
        spec = [x.strip() for x in m.group(1).split("|")]
        if not os.path.exists(os.path.join(ROOT, spec[0])):
            warnings.append("table file missing: " + spec[0])
            return f"**[Table file missing: {spec[0]}]**"
        return csv_table(spec[0], transpose="transpose" in spec[1:])

    def figure_marker(m):
        p = m.group(1).strip()
        if not figures:
            return ""
        if not os.path.exists(os.path.join(ROOT, p)):
            warnings.append("figure file missing: " + p)
            return PAGEBREAK + f"\n**[Figure file missing: {p}]**\n"
        return PAGEBREAK + f"\n![]({os.path.join(ROOT, p)}){{width=100%}}\n"

    text = re.sub(r"<!--\s*TABLE:\s*(.*?)\s*-->", table_marker, text)
    text = re.sub(r"<!--\s*FIGURE:\s*(.*?)\s*-->", figure_marker, text)
    return text.replace("<!-- PAGEBREAK -->", PAGEBREAK)
