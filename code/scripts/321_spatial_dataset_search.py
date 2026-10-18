"""Round 3, WP3 — reproducible search for public spatial transcriptomics of the adult cochlea / inner ear / temporal bone
and of skull or long-bone marrow (NCBI GEO E-utilities; GSE series only).

Each hit is classified from its title / type / organism: spatial modality (Visium, GeoMx, Slide-seq, MERFISH,
Stereo-seq, Xenium, CosMx, 'spatial'), tissue match, developmental stage (embryo / postnatal / developing / otocyst),
disease model (leukaemia, myeloma, tumour, cancer, metastasis, fibrosis …) and organism. Decision: 'used' (GSE312646,
analysed in 252/320), 'candidate' (mouse, adult, non-disease, spatial, matching tissue) or 'excluded' with the reason.
Output (results/r3_spatial/): spatial_dataset_search.csv
"""
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import log, write_csv  # noqa: E402

E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
QUERIES = {
    "cochlea spatial": "cochlea AND (spatial transcriptomics OR Visium OR GeoMx OR MERFISH OR Slide-seq OR Stereo-seq OR Xenium)",
    "inner ear spatial": "inner ear AND (spatial transcriptomics OR Visium OR GeoMx OR MERFISH OR Slide-seq OR Stereo-seq OR Xenium)",
    "temporal bone spatial": "temporal bone AND (spatial OR Visium OR GeoMx)",
    "skull marrow spatial": "(skull OR calvaria) AND bone marrow AND (spatial OR Visium OR MERFISH OR Stereo-seq)",
    "bone marrow spatial (mouse)": "bone marrow AND (Visium OR Slide-seq OR MERFISH OR Stereo-seq OR GeoMx OR Xenium) AND Mus musculus[orgn]",
}
SPATIAL = re.compile(r"spatial|visium|geomx|slide-seq|merfish|stereo-seq|xenium|cosmx", re.I)
TISSUE = re.compile(r"cochle|inner ear|spiral gangl|stria|organ of corti|temporal bone|skull|calvari|bone marrow|marrow", re.I)
DEV = re.compile(r"embryo|embryonic|\bE1\d|postnatal|\bP\d\b|developing|development|otocyst|neonat", re.I)
DIS = re.compile(r"leuk|myeloma|aml|tumou?r|cancer|metasta|fibros|lupus|sepsis|infect|stroke|alzheimer|carcinoma|lymphoma", re.I)


def get(url):
    for proxy in (False, True):
        cmd = (["bash", "-c", f"curl -s --http1.1 -m 60 '{url}'"] if proxy else
               ["curl", "-s", "--http1.1", "-m", "60", url])
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        if out.strip():
            return out
    return ""


rows = []
for qname, q in QUERIES.items():
    ids = re.findall(r"<Id>(\d+)</Id>", get(f"{E}/esearch.fcgi?db=gds&retmax=60&term={q.replace(' ', '+')}"))
    ids = [i for i in ids if i.startswith("200")]
    time.sleep(0.5)
    if not ids:
        continue
    root = ET.fromstring(get(f"{E}/esummary.fcgi?db=gds&id={','.join(ids)}"))
    for doc in root.iter("DocSum"):
        it = {x.get("Name"): (x.text or "") for x in doc.findall("Item")}      # top level only (samples nest Accession items)
        title, typ, taxon = it.get("title", ""), it.get("gdsType", ""), it.get("taxon", "")
        acc = it.get("Accession", "")
        if not acc.startswith("GSE"):
            continue
        text = f"{title} {typ}"
        f = {"spatial": bool(SPATIAL.search(text)), "tissue_match": bool(TISSUE.search(title)),
             "developmental": bool(DEV.search(title)), "disease_model": bool(DIS.search(title)), "mouse": "Mus musculus" in taxon}
        if acc == "GSE312646":
            dec = "used (GeoMx; 252 / 320)"
        elif f["spatial"] and f["tissue_match"] and f["mouse"] and not f["developmental"] and not f["disease_model"]:
            dec = "candidate (manual check)"
        else:
            reasons = [k for k, bad in [("no spatial modality", not f["spatial"]), ("tissue not matched", not f["tissue_match"]),
                                        ("developmental", f["developmental"]), ("disease model", f["disease_model"]),
                                        ("not mouse", not f["mouse"])] if bad]
            dec = "excluded: " + ", ".join(reasons)
        rows.append({"query": qname, "accession": acc, "title": title, "type": typ, "taxon": taxon, "n_samples": it.get("n_samples", ""),
                     "public_date": it.get("PDAT", ""), **f, "decision": dec})
    time.sleep(0.5)
res = pd.DataFrame(rows).drop_duplicates("accession")
write_csv(res.sort_values(["decision", "accession"]), "r3_spatial", "spatial_dataset_search.csv",
          "Systematic GEO search for adult cochlea / temporal bone / marrow spatial transcriptomics with classification and decision",
          required=["query", "accession", "title", "decision"])
log("\n" + res["decision"].value_counts().to_string())
log("\n" + res[~res["decision"].str.startswith("excluded")][["accession", "title", "decision"]].to_string(max_colwidth=90))
