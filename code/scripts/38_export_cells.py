"""Export per-cell counts of candidate genes (Tier-1/2 rhythmic genes + core clock) for the cell-level GLMM
(38_cell_level_glmm.R): <= 300 random cells per cell type × sample (seed 1), raw counts, library size, sample, ZT.
Output: cache/38_cell_level/cells_<tissue>.csv.gz (internal input, not a deliverable).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import H5AD, RESULTS, cache_dir, log, sample_table  # noqa: E402

CLOCK = ["Npas2", "Clock", "Nr1d1", "Nr1d2", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Dbp", "Tef", "Hlf", "Nfil3",
         "Bhlhe40", "Bhlhe41", "Ciart"]
a = ad.read_h5ad(H5AD)
st = sample_table(a.obs).set_index("sample_id")
rng = np.random.default_rng(1)
lib = np.asarray(a.X.sum(1)).ravel()
for tissue, obs_val in [("cochlea", "cochlea"), ("bm", "bone marrow")]:
    t = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv", usecols=["gene", "tier"])
    genes = sorted(set(t.loc[t["tier"] == "tier1", "gene"]) | set(CLOCK))
    genes = [g for g in genes if g in a.var_names]
    m = np.where((a.obs["tissue"] == obs_val).to_numpy())[0]
    obs = a.obs.iloc[m]
    keep = []
    for _, idx in obs.groupby(["batch_id", "cell_type"], observed=True).indices.items():
        keep.append(m[rng.choice(idx, size=min(300, len(idx)), replace=False)])
    keep = np.sort(np.concatenate(keep))
    X = a[keep, genes].X.toarray().astype(np.int32)
    df = pd.DataFrame(X, columns=genes)
    df.insert(0, "lib", lib[keep])
    df.insert(0, "zt", a.obs["batch_id"].iloc[keep].astype(str).map(st["zt"]).to_numpy())
    df.insert(0, "sample_id", a.obs["batch_id"].iloc[keep].astype(str).to_numpy())
    df.insert(0, "cell_type", a.obs["cell_type"].iloc[keep].astype(str).to_numpy())
    df.to_csv(cache_dir("38_cell_level") / f"cells_{tissue}.csv.gz", index=False)
    log(f"{tissue}: {len(keep)} cells × {len(genes)} genes")
