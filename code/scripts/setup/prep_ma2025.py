"""Prepare the user's Ma 2025 upload (2026-10-14) for the 300 → 314 pipeline. Run on the server with python -I.

Input (external/ma2025/): CRA019820_STARsolo_matrices_metadata.zip (STARsolo 2.7.11b, mm10-3.0.0,
10x 3′ v3.1; 10 libraries) and CRA019827_bulk_matrices_metadata.zip (STAR GeneCounts; 12 libraries).
- scRNA: extracts only metadata/, README, per-library Solo.out/Gene/filtered (EmptyDrops_CR cell calls), Summary.csv,
  Features.stats, Barcodes.stats, Log.final.out, per_cell_qc.tsv and REPORT.txt into extracted/ — not the raw whitelist
  matrices, splice-junction tables or the package's helper scripts. Every extracted member is checked against the
  package's MANIFEST.sha256.
- bulk: extracts the package (MANIFEST-checked) and writes CRA019827_bulk_counts_unstranded.tsv (gene symbol × sample;
  unstranded column — forward ≈ reverse assignment in every library; symbols from the package's genes.tsv, duplicate
  symbols summed), the single matrix 314 picks up.
Output: extracted/ + CRA019827_bulk_counts_unstranded.tsv + prep_report.tsv in the same folder.
"""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

D = Path("external/ma2025")
OUT = D / "extracted"
OUT.mkdir(exist_ok=True)
KEEP = [r"/README\.txt$", r"/MANIFEST\.sha256$", r"/metadata/", r"/Solo\.out/Gene/filtered/", r"/Solo\.out/Gene/Summary\.csv$",
        r"/Solo\.out/Gene/Features\.stats$", r"/Solo\.out/Barcodes\.stats$", r"/STARsolo/Log\.final\.out$", r"/per_cell_qc\.tsv$",
        r"/samples/[^/]+/REPORT\.txt$"]
rep = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def extract(zname, keep=None):
    z = zipfile.ZipFile(D / zname)
    names = [n for n in z.namelist() if not n.endswith("/")]
    man = [n for n in names if n.endswith("MANIFEST.sha256")]
    sums = {}
    if man:
        for line in z.read(man[0]).decode().splitlines():
            m = re.match(r"^([0-9a-f]{64})\s+\*?(.+)$", line.strip())
            if m:
                sums[m.group(2).lstrip("./")] = m.group(1)
    sel = [n for n in names if keep is None or any(re.search(k, n) for k in keep)]
    root = Path(man[0]).parent if man else Path("")
    bad, unchecked = 0, 0
    for n in sel:
        z.extract(n, OUT)
        rel = str(Path(n).relative_to(root)) if man and n.startswith(str(root)) else n
        if n.endswith("MANIFEST.sha256"):
            continue
        exp = sums.get(rel) or sums.get(n)
        if exp is None:
            unchecked += 1
        elif sha(OUT / n) != exp:
            bad += 1
    rep.append({"archive": zname, "members": len(names), "extracted": len(sel), "sha256_mismatch": bad, "not_in_manifest": unchecked,
                "manifest_entries": len(sums)})
    if bad:
        sys.exit(f"{zname}: {bad} SHA-256 mismatches")


extract("CRA019820_STARsolo_matrices_metadata.zip", KEEP)
extract("CRA019827_bulk_matrices_metadata.zip")
b = OUT / "CRA019827_bulk"
cnt = pd.read_csv(b / "gene_counts_unstranded.tsv", sep="\t", index_col=0)
gn = pd.read_csv(b / "genes.tsv", sep="\t", index_col=0)["gene_name"]
cnt.index = cnt.index.map(gn)
cnt = cnt[cnt.index.notna()].groupby(level=0).sum()
cnt.index.name = "gene"
cnt.to_csv(D / "CRA019827_bulk_counts_unstranded.tsv", sep="\t")
rep.append({"archive": "CRA019827_bulk_counts_unstranded.tsv", "members": cnt.shape[0], "extracted": cnt.shape[1], "sha256_mismatch": 0,
            "not_in_manifest": 0, "manifest_entries": 0})
pd.DataFrame(rep).to_csv(D / "prep_report.tsv", sep="\t", index=False)
print(pd.DataFrame(rep).to_string(index=False))
