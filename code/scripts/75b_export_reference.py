"""Export a single-cell reference subsample for BayesPrism (<= 400 cells per cell type per tissue, seed 1; raw counts,
protein-coding-like genes only: Gm*/Rik/mt-/ribosomal genes removed). Internal input for 75b_bayesprism.R."""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import H5AD, cache_dir, log  # noqa: E402

a = ad.read_h5ad(H5AD)
g = pd.Index(a.var_names)
keep_g = ~np.asarray(g.str.match(r"^Gm\d") | g.str.endswith("Rik") | g.str.startswith("mt-")
                  | g.str.match(r"^Rp[sl]\d"), dtype=bool)
rng = np.random.default_rng(1)
d = cache_dir("75_bayesprism")
for tissue, val in [("bm", "bone marrow"), ("cochlea", "cochlea")]:
    m = np.where((a.obs["tissue"] == val).to_numpy())[0]
    obs = a.obs.iloc[m]
    idx = []
    for ct, ii in obs.groupby("cell_type", observed=True).indices.items():
        if len(ii) >= 50:
            idx.append(m[rng.choice(ii, size=min(400, len(ii)), replace=False)])
    idx = np.sort(np.concatenate(idx))
    X = a[idx][:, keep_g].X.toarray().astype(np.int32)
    df = pd.DataFrame(X, index=a.obs_names[idx], columns=g[keep_g])
    df = df.loc[:, df.sum(0) > 0]
    df.to_csv(d / f"reference_counts_{tissue}.csv.gz")
    pd.DataFrame({"cell": a.obs_names[idx], "cell_type": a.obs["cell_type"].iloc[idx].astype(str).values}).to_csv(
        d / f"reference_labels_{tissue}.csv", index=False)
    log(f"{tissue}: {len(idx)} cells × {df.shape[1]} genes")
