"""Round 3, WP1 — map a QC'd external dataset onto the atlas gold-standard cochlear labels (lib/refmap.py).

Usage: python scripts/302_reference_mapping.py <key>
Outputs: cache/r3_query/<key>_mapped.h5ad (adds obs label / confidence and obsm X_scanvi_ref);
results/r3_ingest/<key>_label_transfer.csv (per cell type × sample: cells, confident share, agreement, mean probability);
results/r3_ingest/<key>_marker_check.csv (mean scaled expression of our top markers per transferred label)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import refmap  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

KEY = sys.argv[1]
q = ad.read_h5ad(CACHE / "r3_query" / f"{KEY}_qc.h5ad")
res, Z = refmap.map_query(q, batch_key="sample")
for c in res.columns:
    q.obs[c] = res[c].to_numpy()
q.obsm["X_scanvi_ref"] = Z
q.write_h5ad(CACHE / "r3_query" / f"{KEY}_mapped.h5ad", compression="gzip")
t = q.obs.groupby(["sample", "condition", "label"], observed=True).agg(
    n_cells=("label", "size"), frac_confident=("confident", "mean"), frac_agree=("agree", "mean"),
    mean_scanvi_prob=("scanvi_prob", "mean"), mean_knn_vote=("knn_vote", "mean")).reset_index()
t["frac_of_sample"] = t["n_cells"] / t.groupby("sample")["n_cells"].transform("sum")
write_csv(t, "r3_ingest", f"{KEY}_label_transfer.csv",
          f"{KEY}: transferred gold-standard labels per sample (scANVI/scArches consensus with PCA-kNN), confidence",
          required=["sample", "condition", "label", "n_cells", "frac_confident"])
# marker check: our top specific markers of each label, mean z-scored log expression in cells with that label vs others
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 2) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.4)]
top = {c: g.sort_values("scores", ascending=False)["names"].head(15).tolist() for c, g in mk.groupby("cell_type")}
x = q.copy()
sc.pp.normalize_total(x, target_sum=1e4)
sc.pp.log1p(x)
rows = []
for lab, genes in top.items():
    gg = [g for g in genes if g in x.var_names]
    m = (x.obs["label"] == lab).to_numpy()
    if len(gg) < 3 or m.sum() < 10:
        continue
    X = x[:, gg].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    rows.append({"label": lab, "n_cells": int(m.sum()), "n_markers": len(gg), "marker_z_in_label": float(z[m].mean()),
                 "marker_z_in_others": float(z[~m].mean()), "markers": ";".join(gg)})
write_csv(pd.DataFrame(rows), "r3_ingest", f"{KEY}_marker_check.csv",
          f"{KEY}: our label markers are high in the cells transferred to that label (positive control)",
          required=["label", "n_cells", "marker_z_in_label", "marker_z_in_others"])
log(f"{KEY}: mapped; confident {q.obs['confident'].mean():.2f}")
