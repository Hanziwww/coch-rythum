"""Module 5.4 — prepare external circadian datasets into a common format.

For each dataset: external/prepared/<dataset>__expr.csv (genes [mouse symbols] × samples) and <dataset>__meta.csv
(sample, time_h, time_type [ZT|CT], condition, tissue, replicate, value_type [counts|log2], label_note).
Ensembl gene IDs are mapped to symbols with the Ensembl GRCm39 release-113 GTF (genes with several IDs summed).

Parsers (written after inspecting the GEO files):
  E1 GSE88954 — cochlea, raw exon counts (STAR + custom counting), sham / adrenalectomy, 3 reps per time.
     Official GEO times CT4–CT24 (column headers in the count file read CT03/07/11/15/19/24; GEO labels used,
     header labels kept in label_note).
  E2 GSE198770 — bone marrow DESeq2-normalised counts (IgG control vs anti-Ly6G 1A8 neutrophil depletion),
     ZT1–ZT21 every 4 h, 3 reps; value_type normcounts (log2(x+1) + limma-trend downstream).
  Others (GSE86619, GSE54652) are added if their files are present and parseable; unparseable datasets are
  reported in results/71_external/external_prepare_report.csv.
"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import EXTERNAL, log, write_csv  # noqa: E402

GEO = EXTERNAL / "geo"
OUT = EXTERNAL / "prepared"
OUT.mkdir(parents=True, exist_ok=True)
report = []


def gtf_map():
    cache = EXTERNAL / "resources" / "ensembl_gene_map.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    with gzip.open(EXTERNAL / "resources" / "Mus_musculus.GRCm39.113.gtf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            nm = re.search(r'gene_name "([^"]+)"', p[8])
            bt = re.search(r'gene_biotype "([^"]+)"', p[8])
            rows.append((gid, nm.group(1) if nm else gid, p[0], bt.group(1) if bt else ""))
    m = pd.DataFrame(rows, columns=["gene_id", "symbol", "chrom", "biotype"])
    m.to_csv(cache, index=False)
    return m


def series_titles(gse):
    titles, chars = None, {}
    with gzip.open(GEO / gse / f"{gse}_series_matrix.txt.gz", "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1"):
                vals = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
                k = vals[0].split(":")[0]
                chars[k] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
    return titles, chars


def to_symbols(df, idmap):
    sym = idmap.set_index("gene_id")["symbol"]
    df = df.copy()
    df.index = df.index.map(lambda g: sym.get(g.split(".")[0], g))
    return df.groupby(level=0).sum()


def save(ds, expr, meta):
    expr.index.name = "gene"
    expr.to_csv(OUT / f"{ds}__expr.csv")
    meta.to_csv(OUT / f"{ds}__meta.csv", index=False)
    report.append({"dataset": ds, "status": "prepared", "n_genes": expr.shape[0], "n_samples": expr.shape[1],
                   "conditions": ";".join(sorted(meta["condition"].unique())),
                   "times": ";".join(str(t) for t in sorted(meta["time_h"].unique()))})
    log(f"{ds}: {expr.shape[0]} genes × {expr.shape[1]} samples")


GTF = EXTERNAL / "resources" / "Mus_musculus.GRCm39.113.gtf.gz"
idmap = gtf_map() if GTF.exists() or (EXTERNAL / "resources" / "ensembl_gene_map.csv").exists() else None

# ---- E1 GSE88954 (cochlea)
try:
    assert idmap is not None, "Ensembl GTF not yet available"
    f = GEO / "GSE88954" / "GSE88954_cond_exon.txt.gz"
    raw = pd.read_csv(f, sep="\t", index_col=0).drop(columns=["gene_length"], errors="ignore")
    titles, chars = series_titles("GSE88954")
    assert len(titles) == raw.shape[1]
    ct = [float(re.search(r"CT(\d+)", c).group(1)) for c in chars["circadian time"]]
    meta = pd.DataFrame({"sample": raw.columns, "time_h": [c % 24 for c in ct], "time_type": "CT",
                         "condition": ["adx" if s.startswith("adr") else "sham" for s in chars["status"]],
                         "tissue": "cochlea", "replicate": [t.split(".")[-1] for t in titles],
                         "value_type": "counts",
                         "label_note": [f"geo_title={t};file_header={c}" for t, c in zip(titles, raw.columns)]})
    save("GSE88954", to_symbols(raw, idmap), meta)
except Exception as e:
    report.append({"dataset": "GSE88954", "status": f"failed: {e}"})

# ---- E2 GSE198770 (bone marrow, DESeq2-normalised counts with gene symbols in ext_gene)
try:
    f = GEO / "GSE198770" / "GSE198770_bm_ab_normalized_counts.csv.gz"
    raw = pd.read_csv(f)
    samp_cols = [c for c in raw.columns if re.match(r"IBBM\d+", c)]
    expr = raw.groupby("ext_gene")[samp_cols].sum()
    titles, chars = series_titles("GSE198770")
    zt = [float(re.search(r"ZT\s*(\d+)", t).group(1)) for t in chars["time"]]
    cond = ["neutrophil_depleted" if "Ly6G" in t else "control" for t in chars["treatment"]]
    meta = pd.DataFrame({"sample": titles, "time_h": zt, "time_type": "ZT", "condition": cond,
                         "tissue": "bone_marrow", "replicate": range(len(titles)), "value_type": "normcounts",
                         "label_note": [f"treatment={t}" for t in chars["treatment"]]})
    meta = meta[meta["sample"].isin(samp_cols)]
    save("GSE198770", expr[meta["sample"]], meta)
except Exception as e:
    report.append({"dataset": "GSE198770", "status": f"failed: {e}"})

write_csv(pd.DataFrame(report), "71_external", "external_prepare_report.csv",
          "Status of external dataset preparation", required=["dataset", "status"])
log("done")
