"""Module 5.2a — per-sample cell–cell communication with LIANA+ (consensus of CellPhoneDB-, CellChat-, NATMI-,
Connectome-, SingleCellSignalR-type scores; mouse consensus resource), per tissue, by sample (= ZT).

Input: log1p(CP10k) of all genes; groups = gold-standard cell types with >= 10 cells in the sample.
Outputs: cache/63_ccc/liana_<tissue>.parquet (full); results/63_ccc/ccc_scores_by_sample_<tissue>.csv
         (interactions with aggregate rank <= 0.05 in at least one sample, all their samples kept).
"""
import sys
from pathlib import Path

import anndata as ad
import liana as li
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import H5AD, cache_dir, log, sample_table, write_csv  # noqa: E402

MOD = "63_ccc"
a = ad.read_h5ad(H5AD)
st = sample_table(a.obs).set_index("sample_id")
a.obs["sample_id"] = a.obs["batch_id"].astype(str)
lib = np.asarray(a.X.sum(1)).ravel()
X = sp.diags(1e4 / np.maximum(lib, 1)).astype(np.float32) @ a.X.tocsr().astype(np.float32)
X.data = np.log1p(X.data)
a.X = X.tocsr()
for tissue, obs_val in [("cochlea", "cochlea"), ("bm", "bone marrow")]:
    b = a[a.obs["tissue"] == obs_val].copy()
    # drop cell types with < 10 cells in a sample from that sample only
    key = b.obs["sample_id"].astype(str) + "|" + b.obs["cell_type"].astype(str)
    b = b[key.map(key.value_counts()) >= 10].copy()
    b.obs["cell_type"] = b.obs["cell_type"].astype(str).astype("category")
    log(f"{tissue}: {b.n_obs} cells, {b.obs['cell_type'].nunique()} cell types")
    li.mt.rank_aggregate.by_sample(b, groupby="cell_type", sample_key="sample_id", resource_name="mouseconsensus",
                                   expr_prop=0.1, use_raw=False, n_perms=100, verbose=False, key_added="liana_res")
    res = b.uns["liana_res"].copy()
    res["zt"] = res["sample_id"].map(st["zt"])
    res["tissue"] = tissue
    res.to_parquet(cache_dir(MOD) / f"liana_{tissue}.parquet")
    key_cols = ["source", "target", "ligand_complex", "receptor_complex"]
    sig = res[res["magnitude_rank"] <= 0.05][key_cols].drop_duplicates()
    out = res.merge(sig, on=key_cols)
    keep_cols = ["tissue", "sample_id", "zt"] + key_cols + [c for c in ["lr_means", "cellphone_pvals", "expr_prod",
                 "scaled_weight", "lr_logfc", "spec_weight", "lrscore", "lr_probs", "cellchat_pvals",
                 "specificity_rank", "magnitude_rank"] if c in out.columns]
    write_csv(out[keep_cols], MOD, f"ccc_scores_by_sample_{tissue}.csv",
              f"LIANA+ per-sample interaction scores ({tissue}); interactions ranked <= 0.05 in >= 1 sample",
              required=["tissue", "sample_id", "zt", "source", "target", "ligand_complex", "receptor_complex"])
log("done")
