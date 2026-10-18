"""Round 2, WP3 — prepare additional external circadian datasets in the round-1 format
(external/prepared_r2/<dataset>__expr.csv = genes × samples, <dataset>__meta.csv = sample, time_h, time_type,
condition, tissue, replicate, value_type, label_note).

GSE198770 (Vicanolo 2025; LD, ZT; DESeq2-normalised counts → value_type normcounts):
  bone marrow, gut, liver, lung — IgG control vs anti-Ly6G (1A8) neutrophil depletion, ZT1–21 every 4 h, n = 3;
  skin — DT controls (WT + DT), DT-mediated neutrophil depletion (Mrp8-Cre; iDTR + DT) and neutrophil-specific
  Bmal1 deletion (Mrp8-Cre; Bmal1-flox), irregular ZTs.
GSE54650 (CircaDB, Zhang 2014; DD, CT18–64 every 2 h, one mouse pool per time, 12 organs; MoGene 1.0 ST raw
  intensities → log2; probes → symbols with GPL6246, highest-mean probe per symbol).
GSE25585 (Keller 2009; DD, peritoneal macrophages CD11b+, CT0–20 every 4 h over two days, pools of 3 mice; log2).
Outputs: external/prepared_r2/*; results/r2_external/external_prepare_report.csv
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
OUT = EXTERNAL / "prepared_r2"
OUT.mkdir(parents=True, exist_ok=True)
gmap = pd.read_csv(EXTERNAL / "resources/ensembl_gene_map.csv")
ens2sym = dict(zip(gmap["gene_id"], gmap["symbol"]))
report = []


def series_table(path):
    """Sample table (title, GSM, source, characteristics as key: value columns) from a GEO series matrix."""
    rows = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith(("!Sample_title", "!Sample_geo_accession", "!Sample_source_name_ch1",
                                "!Sample_characteristics_ch1")):
                f = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
                rows.setdefault(f[0], []).append(f[1:])
    df = pd.DataFrame({"title": rows["!Sample_title"][0], "gsm": rows["!Sample_geo_accession"][0],
                       "source": rows["!Sample_source_name_ch1"][0]})
    for vals in rows.get("!Sample_characteristics_ch1", []):
        for i, v in enumerate(vals):
            if ":" in v:
                k, x = v.split(":", 1)
                df.loc[i, k.strip().lower()] = x.strip()
    return df


def series_data(path):
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
        d = pd.read_csv(fh, sep="\t", comment="!", index_col=0)
    return d


def save(name, expr, meta, note):
    expr.index.name = "gene"
    expr.to_csv(OUT / f"{name}__expr.csv")
    meta.to_csv(OUT / f"{name}__meta.csv", index=False)
    report.append({"dataset": name, "n_genes": expr.shape[0], "n_samples": expr.shape[1],
                   "tissues": ";".join(sorted(meta["tissue"].unique())),
                   "conditions": ";".join(sorted(meta["condition"].unique())),
                   "time_type": meta["time_type"].iloc[0], "value_type": meta["value_type"].iloc[0], "note": note})
    log(f"{name}: {expr.shape[0]} genes × {expr.shape[1]} samples; conditions {sorted(meta['condition'].unique())}")


# ---- GSE198770: five tissues
st = series_table(GEO / "GSE198770/GSE198770_series_matrix.txt.gz")
st["zt"] = st["time"].str.extract(r"ZT\s*(\d+)")[0].astype(float) % 24


def cond198(r):
    g, t, m = str(r.get("genotype", "")), str(r.get("treatment", "")), str(r.get("mouse", ""))
    if "Bmal1" in g:
        return "neutrophil_Bmal1_KO"
    if "iDTR" in g:
        return "neutrophil_depleted_DT"
    if "Ly6G" in t:
        return "neutrophil_depleted"
    return "control"


st["condition"] = st.apply(cond198, axis=1)
for tissue, fn in [("bone_marrow", "bm_ab"), ("gut", "gut_ab"), ("liver", "liver_ab"), ("lung", "lung_ab"),
                   ("skin", "skin_dt")]:
    d = pd.read_csv(GEO / f"GSE198770/GSE198770_{fn}_normalized_counts.csv.gz", index_col=0)
    d = d[[c for c in d.columns if c not in ("Row.names", "ext_gene", "mgi_description")]]
    titles = set(st["title"])
    colmap = {}
    for c in d.columns:
        if c in titles:
            colmap[c] = c
        elif re.fullmatch(r"LIV\d+", c) and f"LIVER{c[3:]}" in titles:
            colmap[c] = f"LIVER{c[3:]}"
    d = d[list(colmap)].rename(columns=colmap)
    d["symbol"] = d.index.map(ens2sym)
    d = d.dropna(subset=["symbol"]).groupby("symbol").sum()
    m = st.set_index("title").loc[d.columns]
    meta = pd.DataFrame({"sample": d.columns, "time_h": m["zt"].to_numpy(), "time_type": "ZT",
                         "condition": m["condition"].to_numpy(), "tissue": tissue,
                         "replicate": m.groupby(["zt", "condition"]).cumcount().to_numpy() + 1,
                         "value_type": "normcounts", "label_note": "GEO titles; liver columns LIVnn = titles LIVERnn"})
    save(f"GSE198770_{tissue}", d, meta, "DESeq2-normalised counts; Ensembl → symbol (GRCm39 GTF), summed")

# ---- GPL6246 probe → symbol
ann = pd.read_csv(EXTERNAL / "resources/GPL6246.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"],
                  dtype={"ID": str}).dropna()
ann = ann[~ann["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]


def collapse(expr):
    expr.index = expr.index.astype(str)
    expr = expr[expr.index.isin(ann.index)]
    expr["symbol"] = ann.loc[expr.index].to_numpy()
    expr["mean"] = expr.drop(columns="symbol").mean(1)
    expr = expr.sort_values("mean", ascending=False).drop_duplicates("symbol").set_index("symbol")
    return expr.drop(columns="mean")


# ---- GSE54650 CircaDB (12 organs, DD)
st = series_table(GEO / "GSE54650/GSE54650_series_matrix.txt.gz")
d = collapse(np.log2(series_data(GEO / "GSE54650/GSE54650_series_matrix.txt.gz").clip(lower=1)))
st["ct"] = st["title"].str.extract(r"CT(\d+)")[0].astype(float)
m = st.set_index("gsm").loc[d.columns]
meta = pd.DataFrame({"sample": d.columns, "time_h": (m["ct"] % 24).to_numpy(), "time_type": "CT",
                     "condition": "WT", "tissue": m["source"].str.replace(" ", "_").to_numpy(),
                     "replicate": (m["ct"] >= 42).astype(int).to_numpy() + 1, "value_type": "log2",
                     "label_note": "CT18–64 every 2 h; replicate = cycle (CT18–40 vs CT42–64)"})
save("GSE54650", d, meta, "MoGene 1.0 ST intensities → log2; highest-mean probe per symbol")

# ---- GSE25585 peritoneal macrophages (DD)
st = series_table(GEO / "GSE25585/GSE25585_series_matrix.txt.gz")
d = collapse(series_data(GEO / "GSE25585/GSE25585_series_matrix.txt.gz"))
st["ct"] = st["title"].str.extract(r"CT(\d+)")[0].astype(float)
st["day"] = np.where(st["title"].str.contains("second"), 2, 1)
m = st.set_index("gsm").loc[d.columns]
meta = pd.DataFrame({"sample": d.columns, "time_h": m["ct"].to_numpy(), "time_type": "CT", "condition": "WT",
                     "tissue": "peritoneal_macrophage", "replicate": m["day"].to_numpy(), "value_type": "log2",
                     "label_note": "CT0–20 every 4 h on two consecutive days (pools of 3 mice)"})
save("GSE25585", d, meta, "MoGene 1.0 ST log2 (as deposited); highest-mean probe per symbol")

write_csv(pd.DataFrame(report), "r2_external", "external_prepare_report.csv",
          "Round-2 external datasets prepared for rhythm analysis", required=["dataset", "n_genes", "n_samples"])
log("done")
