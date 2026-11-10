"""Fetch PubMed records for every PMID in references.tsv and cache them as JSON.

Usage: python3 -I manuscript/fetch_refs.py manuscript/references.tsv manuscript/refs_cache.json
Only NCBI E-utilities are contacted; no email or API key is sent.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

tsv, out = sys.argv[1], sys.argv[2]
keys = {}
for line in open(tsv, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 2 and parts[1].strip().isdigit():
        keys[parts[0].strip()] = parts[1].strip()
cache = json.load(open(out)) if os.path.exists(out) else {}
todo = sorted({p for p in keys.values() if p not in cache})


def text(el, path, default=""):
    x = el.find(path)
    return "".join(x.itertext()).strip() if x is not None else default


for i in range(0, len(todo), 40):
    batch = todo[i:i + 40]
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(batch))
    for attempt in range(4):
        try:
            if attempt < 2:
                with urllib.request.urlopen(url, timeout=60) as r:
                    root = ET.fromstring(r.read())
            else:   # some proxies truncate urllib reads (IncompleteRead); curl handles them
                root = ET.fromstring(subprocess.run(["curl", "-s", "--max-time", "90", url], check=True,
                                                    capture_output=True).stdout)
            break
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, e, flush=True)
            time.sleep(3 * (attempt + 1))
    else:
        sys.exit("efetch failed")
    for art in root.findall(".//PubmedArticle"):
        pmid = text(art, ".//MedlineCitation/PMID")
        a = art.find(".//Article")
        authors = []
        for au in a.findall(".//AuthorList/Author"):
            if au.find("LastName") is not None:
                authors.append(f"{text(au, 'LastName')} {text(au, 'Initials')}".strip())
            elif au.find("CollectiveName") is not None:
                authors.append(text(au, "CollectiveName"))
        year = text(a, ".//Journal/JournalIssue/PubDate/Year") or text(a, ".//Journal/JournalIssue/PubDate/MedlineDate")[:4]
        if not year:
            year = text(art, ".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year")
        eloc = {e.get("EIdType"): "".join(e.itertext()) for e in a.findall("ELocationID")}
        ids = {e.get("IdType"): "".join(e.itertext()) for e in art.findall(".//PubmedData/ArticleIdList/ArticleId")}
        cache[pmid] = dict(
            pmid=pmid, authors=authors, title=text(a, "ArticleTitle"),
            journal=text(art, ".//MedlineJournalInfo/MedlineTA") or text(a, ".//Journal/ISOAbbreviation"),
            year=year, volume=text(a, ".//Journal/JournalIssue/Volume"), issue=text(a, ".//Journal/JournalIssue/Issue"),
            pages=text(a, ".//Pagination/MedlinePgn"), elocation=eloc.get("pii") or "",
            doi=eloc.get("doi") or ids.get("doi", ""))
    time.sleep(0.4)
json.dump(cache, open(out, "w"), indent=1, ensure_ascii=False)
missing = [k for k, p in keys.items() if p not in cache]
print(f"{len(cache)} records cached; missing: {missing}")
