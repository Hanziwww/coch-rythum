"""Round 3, WP0/WP1 — ingest external cochlear single-cell count matrices into one AnnData per dataset (raw counts,
gene symbols as var_names, obs: dataset, sample, condition, group).

Usage: python scripts/300_external_sc_ingest.py <key>     key ∈ {gse167078, gse346691, ma2025, gse274279, gse234926,
                                                                  gse283534, gse135913}
- gse167078 (Milon 2021, Cell Rep, doi 10.1016/j.celrep.2021.109758): CD45+ cochlear cells, one 10x v2 library
  hashed with TotalSeq-A 0301 (sham), 0302 (3 d), 0303 (7 d), 0304 (14 d after 6–12 kHz / 110 dB noise for 2 h from
  20:00); hashtags demultiplexed here (CLR per hashtag across cells; singlet = top CLR exceeding the second by >= 0.5).
- gse346691: 7 whole-cochlea 10x samples — WT noise (110 dB; 3), WT control (2), Spp1-KO noise (1), Ccr2-KO noise (1).
- ma2025 (Ma 2025, Neurosci Bull, doi 10.1007/s12264-025-01440-1; CRA019820): matrices quantified and uploaded by the
  user — searched under external/ma2025 and external/ma2025; per sample a 10x mtx folder, .h5 or
  .h5ad whose name contains the sample name of data/CRA019820.xlsx (C1_ctrl … N3_Ne3d); raw (unfiltered) matrices are
  flagged for cell calling in 301. Groups: control (C), day noise ZT2–4 (D), night noise ZT14–16 (N); 3 d after noise.
  The user's upload is STARsolo 2.7.11b (mm10-3.0.0, 10x 3′ v3.1): its Solo.out/Gene/filtered matrices (EmptyDrops_CR
  cell calls, equivalent to 301 step 1) are ingested; the raw whitelist matrices are not re-called.
- round 4 (WP-E, atlas concordance): gse274279 (snRNA cochlea 3 / 12 / 24 months; utricle excluded), gse234926 (P90
  cochlear-duct Fbxo2+ nuclei, M / F; raw matrices), gse283534 (10-week cochlea, 6 replicates; vestibule excluded),
  gse135913 (adult mouse cochlea, epithelial-sorted and negative fractions; Cell Ranger v2 genes.tsv).
Outputs: cache/r3_query/<key>.h5ad; results/r3_ingest/<key>_ingest_report.csv
"""
import hashlib
import os
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, EXTERNAL, ROOT, cache_dir, log, write_csv  # noqa: E402

KEY = sys.argv[1]
OUT = cache_dir("r3_query")
GEO = EXTERNAL / "geo"
report = []


def read_mtx_triplet(prefix):
    """10x triplet with a file prefix (…_barcodes.tsv.gz, …_features.tsv.gz, …_matrix.mtx.gz) → AnnData (cells × genes)."""
    m = scipy.io.mmread(f"{prefix}matrix.mtx.gz").tocsr().T.tocsr()
    ff = f"{prefix}features.tsv.gz" if Path(f"{prefix}features.tsv.gz").exists() else f"{prefix}genes.tsv.gz"   # Cell Ranger v2: genes.tsv
    f = pd.read_csv(ff, sep="\t", header=None)
    b = pd.read_csv(f"{prefix}barcodes.tsv.gz", sep="\t", header=None)[0].to_numpy()
    x = ad.AnnData(m.astype(np.float32), obs=pd.DataFrame(index=b),
                   var=pd.DataFrame({"gene_ids": f[0].to_numpy(), "feature_type": f[2].to_numpy() if f.shape[1] > 2 else "Gene Expression"},
                                    index=f[1].astype(str).to_numpy()))
    return x


def finish(x):
    x = x[:, x.var["feature_type"].astype(str).str.contains("Gene")].copy() if "feature_type" in x.var else x
    x.var_names_make_unique()
    x.obs_names_make_unique()
    return x


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if KEY == "gse167078":
    d = GEO / "GSE167078/raw"
    rna = read_mtx_triplet(str(d / "GSM5092462_RNA_"))
    hto = read_mtx_triplet(str(d / "GSM5092463_HTO_"))
    H = hto[:, hto.var["feature_type"].astype(str).str.contains("Antibody")]
    H = pd.DataFrame(H.X.toarray(), index=H.obs_names, columns=H.var_names)
    clr = np.log1p(H).sub(np.log1p(H).mean(0), axis=1)          # CLR across cells per hashtag
    srt = np.sort(clr.to_numpy(), 1)
    top = clr.idxmax(1)
    singlet = (srt[:, -1] - srt[:, -2]) >= 0.5
    cond = {"TotalSeq-A_0301": "sham", "TotalSeq-A_0302": "noise_3d", "TotalSeq-A_0303": "noise_7d", "TotalSeq-A_0304": "noise_14d"}
    x = finish(rna)
    x.obs["hashtag"] = top.reindex(x.obs_names).to_numpy()
    x.obs["hto_singlet"] = pd.Series(singlet, index=H.index).reindex(x.obs_names).to_numpy()
    x = x[x.obs["hto_singlet"].astype(bool)].copy()
    x.obs["condition"] = x.obs["hashtag"].map(cond).to_numpy()
    x.obs["sample"] = x.obs["condition"]
    x.obs["group"] = x.obs["condition"]
    x.uns["matrix_level"] = "filtered"
    for c, n in x.obs["condition"].value_counts().items():
        report.append({"dataset": KEY, "sample": c, "n_barcodes": int(n), "matrix_level": "filtered (Cell Ranger)",
                       "note": f"hashtag singlets; {int((~singlet).sum())} of {len(singlet)} barcodes not assigned"})
elif KEY == "gse346691":
    d = GEO / "GSE346691/raw"
    if not d.exists():
        d.mkdir(parents=True)
        import tarfile
        tarfile.open(GEO / "GSE346691/GSE346691_RAW.tar").extractall(d)
    cond = {"Spp1KO_110dB": "Spp1_KO_noise", "WT_110dB": "WT_noise", "WT": "WT_noise", "WT_NE3": "WT_noise",
            "WT_CON2": "WT_control", "WT_CON3": "WT_control", "Ccr2_NE": "Ccr2_KO_noise"}
    xs = []
    for f in sorted(d.glob("*_matrix.mtx.gz")):
        prefix = str(f).replace("matrix.mtx.gz", "")
        name = re.sub(r"^GSM\d+_", "", Path(prefix).name.rstrip("_"))
        x = finish(read_mtx_triplet(prefix))
        x.obs["sample"], x.obs["condition"] = name, cond[name]
        x.obs_names = [f"{name}:{b}" for b in x.obs_names]
        xs.append(x)
        report.append({"dataset": KEY, "sample": name, "n_barcodes": x.n_obs, "matrix_level": "filtered (as deposited)",
                       "note": cond[name]})
    x = ad.concat(xs, join="inner", merge="same")
    x.obs["group"] = x.obs["condition"]
    x.uns["matrix_level"] = "filtered"
elif KEY in ("ma2025", "ma2025dry"):                # ma2025dry: pipeline dry run on stand-in matrices under $MA_ROOT
    meta = pd.read_excel(ROOT / "data/CRA019820.xlsx", sheet_name="Sample")[["Sample name"]]
    names = meta["Sample name"].astype(str).tolist()
    roots = [Path(os.environ["MA_ROOT"])] if KEY == "ma2025dry" else [Path("external/ma2025"), EXTERNAL / "ma2025"]
    cands = [p for r in roots if r.exists() for p in r.rglob("*") if p.is_dir() or p.suffix in (".h5", ".h5ad")]
    xs = []
    for nm in names:
        hits = [p for p in cands if nm.lower() in str(p).lower()]
        mtx = [p for p in hits if p.is_dir() and (any(p.glob("matrix.mtx*")))]
        # STARsolo output (user upload): Solo.out/Gene/filtered = EmptyDrops_CR cell calls (the Cell Ranger emptyDrops, i.e.
        # 301 step 1); its raw/ folder holds the whole 3′ v3.1 whitelist (~6.8 M barcodes) and is not re-called
        solo = [p for p in mtx if "solo.out" in str(p).lower() and p.name.lower() == "filtered"]
        raw_dirs = [p for p in mtx if "raw" in p.name.lower() and "solo.out" not in str(p).lower()]
        h5 = [p for p in hits if p.suffix == ".h5"]
        h5ad_ = [p for p in hits if p.suffix == ".h5ad"]
        src = (solo or raw_dirs or mtx or [p for p in h5 if "raw" in p.name.lower()] or h5 or h5ad_ or [None])[0]
        if src is None:
            report.append({"dataset": KEY, "sample": nm, "n_barcodes": 0, "matrix_level": "missing", "note": "no file found"})
            continue
        x = (sc.read_10x_mtx(src, var_names="gene_symbols", make_unique=True, compressed=not (src / "matrix.mtx").exists()) if src.is_dir() else
             sc.read_10x_h5(src) if src.suffix == ".h5" else ad.read_h5ad(src))
        x.var_names_make_unique()
        level = "filtered" if src in solo else ("raw" if x.n_obs > 50000 or "raw" in src.name.lower() else "filtered")
        grp = {"C": "control", "D": "day_noise", "N": "night_noise"}[nm[0]]
        x.obs["sample"], x.obs["condition"], x.obs["group"], x.obs["matrix_level"] = nm, grp, grp, level
        x.obs_names = [f"{nm}:{b}" for b in x.obs_names]
        xs.append(x)
        report.append({"dataset": KEY, "sample": nm, "n_barcodes": x.n_obs, "matrix_level": level, "note": str(src),
                       "md5": md5(src) if src.is_file() else ""})
    if not xs:
        write_csv(pd.DataFrame(report), "r3_ingest", f"{KEY}_ingest_report.csv", "Ma 2025 ingestion: files not yet uploaded",
                  required=["dataset", "sample", "matrix_level"])
        sys.exit("no Ma 2025 matrices found yet")
    x = ad.concat(xs, join="inner", merge="same")
    x.uns["matrix_level"] = "raw" if (x.obs["matrix_level"] == "raw").any() else "filtered"
elif KEY in ("gse274279", "gse234926", "gse283534", "gse135913"):    # round 4: external adult cochlear atlases (WP-E)
    import tarfile
    acc = KEY.upper()
    tarf = GEO / acc / f"{acc}_RAW.tar"
    d = GEO / acc / "raw"
    if tarf.exists() and not d.exists():
        d.mkdir(parents=True)
        tarfile.open(tarf).extractall(d)
    src = d if d.exists() else GEO / acc
    PAT = {"gse274279": r"^GSM\d+_((\d+M)_Cochlea(?:_BU)?)_$",           # cochlea only (utricle excluded)
           "gse234926": r"^GSM\d+_(P90_Fbxo2_([MF]))_$",
           "gse283534": r"^GSM\d+_(Cochlea_10weeks_Bio_rep\d)_$",          # vestibule excluded
           "gse135913": r"^GSM\d+_(m_adult_(epi|neg)2?)_$"}[KEY]
    xs = []
    for f in sorted(src.glob("*matrix.mtx.gz")):
        prefix = str(f).replace("matrix.mtx.gz", "")
        m_ = re.match(PAT, Path(prefix).name)
        if not m_:
            continue
        name = m_.group(1)
        cond = {"gse274279": m_.group(2) if m_.lastindex >= 2 else "", "gse234926": "P90", "gse283534": "10wk",
                "gse135913": f"adult_{m_.group(2)}" if m_.lastindex >= 2 else "adult"}[KEY]
        x = finish(read_mtx_triplet(prefix))
        level = "raw" if x.n_obs > 50000 else "filtered"
        x.obs["sample"], x.obs["condition"], x.obs["group"], x.obs["matrix_level"] = name, cond, cond, level
        x.obs_names = [f"{name}:{b}" for b in x.obs_names]
        xs.append(x)
        report.append({"dataset": KEY, "sample": name, "n_barcodes": x.n_obs, "matrix_level": level, "note": cond})
    x = ad.concat(xs, join="inner", merge="same")
    x.uns["matrix_level"] = "raw" if (x.obs["matrix_level"] == "raw").any() else "filtered"
else:
    sys.exit(f"unknown key {KEY}")

x.obs["dataset"] = KEY
x.X = sp.csr_matrix(x.X).astype(np.float32)
x.write_h5ad(OUT / f"{KEY}.h5ad", compression="gzip")
rep = pd.DataFrame(report)
rep["n_genes"] = x.n_vars
write_csv(rep, "r3_ingest", f"{KEY}_ingest_report.csv", f"Ingestion report for {KEY}: samples, barcodes, matrix level",
          required=["dataset", "sample", "n_barcodes", "matrix_level"])
log(f"{KEY}: {x.n_obs} barcodes × {x.n_vars} genes; " + ", ".join(f"{k} {v}" for k, v in x.obs["condition"].value_counts().items()))
