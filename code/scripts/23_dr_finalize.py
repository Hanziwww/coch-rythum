"""Module 1 — final embeddings: pre-registered composite selection, then neighbours/UMAP on all cells.

Selection (from results/20_dr/dr_benchmark_metrics_<family>.csv):
  per-tissue = 0.7·bio + 0.3·time;  joint = 0.5·bio + 0.2·platform mixing + 0.3·time;
  embeddings whose time-signal ratio < 0.8 (lost > 20 % of the time-of-day signal) are rejected.
Outputs: results/20_dr/dr_benchmark_summary.csv, embedding_<family>_latent.csv, umap_<family>.csv,
         neighbors_summary.csv; cache/20_dr/final_<family>.h5ad (obs, latent, UMAP, kNN graph) for Modules 4–5.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, cache_dir, log, write_csv  # noqa: E402

rows, nb_rows = [], []
for fam in ["cochlea", "bm", "joint"]:
    m = pd.read_csv(RESULTS / f"20_dr/dr_benchmark_metrics_{fam}.csv")
    w = m.pivot_table(index="embedding", columns="metric", values="value")
    w = w[~w.index.str.startswith(("A__", "ref__"))]
    if "score_time" not in w:
        raise SystemExit(f"{fam}: time ratios missing — rerun 22_dr_metrics.py with a baseline")
    if fam == "joint":
        w["composite"] = 0.5 * w["score_bio"] + 0.2 * w["score_platform"] + 0.3 * w["score_time"]
    else:
        w["composite"] = 0.7 * w["score_bio"] + 0.3 * w["score_time"]
    w["rejected_time_loss"] = w["score_time"] < 0.8
    w["rank"] = w["composite"].where(~w["rejected_time_loss"]).rank(ascending=False)
    w = w.sort_values(["rejected_time_loss", "composite"], ascending=[True, False])
    w.insert(0, "family", fam)
    rows.append(w.reset_index())
    best = w.index[0]
    log(f"{fam}: selected {best} (composite {w.loc[best, 'composite']:.3f})")
    root = cache_dir(f"20_dr/{fam}")
    obs = pd.read_parquet(root / "obs.parquet")
    Z = np.load(root / "emb" / f"{best}.npy")
    a = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    a.obsm["X_emb"] = Z
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=30)
    sc.tl.umap(a, random_state=0)
    a.uns["selected_embedding"] = best
    a.write_h5ad(root.parent / f"final_{fam}.h5ad")
    lat = pd.DataFrame(Z, index=obs.index, columns=[f"dim{i + 1}" for i in range(Z.shape[1])])
    lat.index.name = "cell_id"
    write_csv(lat.reset_index(), "20_dr", f"embedding_{fam}_latent.csv", f"Selected latent embedding ({best}), {fam}",
              required=["cell_id"])
    um = pd.DataFrame(a.obsm["X_umap"], index=obs.index, columns=["umap1", "umap2"])
    um.index.name = "cell_id"
    write_csv(um.reset_index(), "20_dr", f"umap_{fam}.csv", f"UMAP of the selected embedding, {fam}",
              required=["cell_id", "umap1", "umap2"])
    d = a.obsp["distances"]
    nb_rows.append({"family": fam, "embedding": best, "n_cells": a.n_obs, "dims": Z.shape[1], "k": 30,
                    "mean_knn_distance": float(d.sum() / d.nnz),
                    "n_connected_components": int(connected_components(a.obsp["connectivities"])[0])})
summary = pd.concat(rows, ignore_index=True)
write_csv(summary, "20_dr", "dr_benchmark_summary.csv",
          "Composite scores, time-loss rejection and ranks of all Stage B/C embeddings per family",
          required=["family", "embedding", "score_bio", "score_time", "composite", "rank"])
write_csv(pd.DataFrame(nb_rows), "20_dr", "neighbors_summary.csv", "kNN graph summary of the selected embeddings",
          required=["family", "embedding", "k"])
log("done")
