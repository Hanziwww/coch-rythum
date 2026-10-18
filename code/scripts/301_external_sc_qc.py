"""Round 3, WP1 — cell calling and QC of an ingested external dataset (no ambient-RNA correction, as for the atlas).

Usage: python scripts/301_external_sc_qc.py <key>
1. Raw (unfiltered) matrices only: DropletUtils emptyDrops per sample (scripts/301b_emptydrops.R; lower = 100 UMIs,
   FDR < 0.001, plus barcodes above the knee retained by emptyDrops' default).
2. Per-sample QC: >= 200 genes and >= 500 UMIs; log UMIs and log genes within median ± 3 MAD; % mitochondrial
   <= min(25, median + 3 MAD).
3. Doublets: scrublet per sample (scanpy), predicted doublets removed.
Outputs: cache/r3_query/<key>_qc.h5ad; results/r3_ingest/<key>_cell_qc.csv
"""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, ROOT, cache_dir, log, write_csv  # noqa: E402

KEY = sys.argv[1]
q = ad.read_h5ad(CACHE / "r3_query" / f"{KEY}.h5ad")
rows = []
if q.uns.get("matrix_level", "filtered") == "raw" or ("matrix_level" in q.obs and (q.obs["matrix_level"] == "raw").any()):
    tmp = cache_dir(f"r3_query/{KEY}_emptydrops")
    keep = []
    for s in q.obs["sample"].unique():
        x = q[q.obs["sample"] == s]
        if "matrix_level" in x.obs and (x.obs["matrix_level"] != "raw").all():
            keep += list(x.obs_names)
            continue
        d = tmp / str(s)
        d.mkdir(parents=True, exist_ok=True)
        scipy.io.mmwrite(str(d / "matrix.mtx"), x.X.T.tocsr().astype(np.int64))
        pd.Series(x.obs_names).to_csv(d / "barcodes.tsv", index=False, header=False)
        pd.Series(x.var_names).to_csv(d / "genes.tsv", index=False, header=False)
        subprocess.run(["bash", str(ROOT / "scripts/setup/rjob.sh"), str(ROOT / "scripts/301b_emptydrops.R"), str(d)], check=True)
        cells = pd.read_csv(d / "cells.csv")["barcode"].tolist()
        keep += cells
        rows.append({"sample": s, "step": "emptyDrops", "n_cells": len(cells), "n_barcodes_in": x.n_obs})
    q = q[q.obs_names.isin(keep)].copy()
q.var["mt"] = q.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(q, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)
ok = np.ones(q.n_obs, bool)
for s, idx in q.obs.groupby("sample", observed=True).indices.items():
    o = q.obs.iloc[idx]
    good = (o["n_genes_by_counts"] >= 200) & (o["total_counts"] >= 500)
    for col in ["log1p_total_counts", "log1p_n_genes_by_counts"]:
        med = o[col].median()
        mad = (o[col] - med).abs().median() * 1.4826
        good &= (o[col] - med).abs() <= 3 * mad
    mt_med = o["pct_counts_mt"].median()
    mt_mad = (o["pct_counts_mt"] - mt_med).abs().median() * 1.4826
    good &= o["pct_counts_mt"] <= min(25.0, mt_med + 3 * mt_mad)
    ok[idx] = good.to_numpy()
    rows.append({"sample": s, "step": "QC thresholds", "n_cells": int(good.sum()), "n_barcodes_in": len(o),
                 "median_umis": float(o["total_counts"].median()), "median_genes": float(o["n_genes_by_counts"].median()),
                 "median_pct_mt": float(mt_med)})
q = q[ok].copy()
sc.pp.scrublet(q, batch_key="sample", random_state=0)
for s, o in q.obs.groupby("sample", observed=True):
    rows.append({"sample": s, "step": "scrublet singlets", "n_cells": int((~o["predicted_doublet"]).sum()), "n_barcodes_in": len(o)})
q = q[~q.obs["predicted_doublet"].to_numpy()].copy()
q.write_h5ad(CACHE / "r3_query" / f"{KEY}_qc.h5ad", compression="gzip")
rep = pd.DataFrame(rows)
rep.insert(0, "dataset", KEY)
write_csv(rep, "r3_ingest", f"{KEY}_cell_qc.csv", f"Cell calling / QC / doublet filtering for {KEY} per sample",
          required=["dataset", "sample", "step", "n_cells"])
log(f"{KEY}: {q.n_obs} cells after QC")
