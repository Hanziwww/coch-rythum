"""Module 2.1 — pseudobulk count matrices per tissue × gold-standard group × sample.

Groups: cell_type (primary resolution) and cell_class (coarse resolution).
Inclusion: primary if >= min_cells_primary cells in all 6 samples of the tissue; flagged if >= min_cells_flag
in all 6 but below primary somewhere; otherwise excluded from rhythm analyses (still counted in the design table).

Outputs:
  results/30_pseudobulk/pseudobulk_design.csv            one row per tissue × level × group × sample
  results/30_pseudobulk/groups_included.csv              inclusion status per tissue × level × group
  results/30_pseudobulk/counts/{tissue}__{level}__{group}.csv   genes × 6 samples (raw summed counts), included groups
  cache/30_pseudobulk/{tissue}__{level}.parquet           all groups, genes × (group|sample) columns (internal)
"""
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CFG, H5AD, LABEL_COLS, ZT_VALUES, cache_dir, log, result_dir, sample_table, write_csv  # noqa: E402
from lib.pseudobulk import by_sample_and_label  # noqa: E402

MOD = "30_pseudobulk"
MIN_P = CFG["pseudobulk"]["min_cells_primary"]
MIN_F = CFG["pseudobulk"]["min_cells_flag"]


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


log(f"reading {H5AD}")
adata = ad.read_h5ad(H5AD)
samples = sample_table(adata.obs)
label_hash = pd.util.hash_pandas_object(adata.obs[LABEL_COLS].astype(str), index=True).sum()
total_counts = float(adata.X.sum())

design_rows, incl_rows = [], []
for level in ["cell_type", "cell_class"]:
    counts, meta = by_sample_and_label(adata, label=level, sample="batch_id")
    assert np.isclose(counts.sum(), total_counts, rtol=1e-6), "pseudobulk sums must equal raw counts"
    meta = meta.rename(columns={"batch_id": "sample_id", level: "group"}).merge(samples[["sample_id", "tissue", "zt"]],
                                                                                 on="sample_id")
    for tissue, tsamp in samples.groupby("tissue"):
        tmeta = meta[meta["tissue"] == tissue]
        sample_order = tsamp.sort_values("zt")["sample_id"].tolist()
        cols, mats = [], []
        for group, gm in tmeta.groupby("group"):
            n_by_sample = gm.set_index("sample_id")["n_cells"].reindex(sample_order).fillna(0).astype(int)
            min_n = int(n_by_sample.min())
            status = "primary" if min_n >= MIN_P else ("flagged" if min_n >= MIN_F else "excluded")
            incl_rows.append({"tissue": tissue, "level": level, "group": group, "group_safe": safe(group),
                              "min_cells_per_sample": min_n, "total_cells": int(n_by_sample.sum()),
                              "status": status})
            for sid in sample_order:
                row = gm[gm["sample_id"] == sid]
                design_rows.append({"tissue": tissue, "level": level, "group": group, "sample_id": sid,
                                    "clock_time": float(tsamp.set_index("sample_id").loc[sid, "clock_time"]),
                                    "zt": float(tsamp.set_index("sample_id").loc[sid, "zt"]),
                                    "n_cells": int(row["n_cells"].iloc[0]) if len(row) else 0,
                                    "lib_size": float(row["total_counts"].iloc[0]) if len(row) else 0.0,
                                    "status": status})
            # matrix genes × samples (zeros where a group is absent from a sample)
            mat = np.zeros((adata.n_vars, len(sample_order)), dtype=np.float64)
            for j, sid in enumerate(sample_order):
                idx = gm.index[gm["sample_id"] == sid]
                if len(idx):
                    mat[:, j] = counts[idx[0]].toarray().ravel()
            cols += [f"{group}|{sid}" for sid in sample_order]
            mats.append(mat)
            if status != "excluded":
                df = pd.DataFrame(mat.astype(np.int64), index=adata.var_names,
                                  columns=[f"ZT{int(z):02d}_{sid}" for z, sid in
                                           zip(sorted(tsamp["zt"]), sample_order)])
                df.index.name = "gene"
                write_csv(df, MOD, f"counts/{tissue}__{level}__{safe(group)}.csv",
                          f"Pseudobulk raw counts ({tissue}, {level}={group}); columns ordered by ZT", index=True)
        big = pd.DataFrame(np.hstack(mats), index=adata.var_names, columns=cols)
        big.to_parquet(cache_dir(MOD) / f"{tissue}__{level}.parquet")
        log(f"{tissue} {level}: {len(mats)} groups cached")

design = pd.DataFrame(design_rows)
incl = pd.DataFrame(incl_rows).sort_values(["level", "tissue", "status", "group"])
assert set(design["zt"].unique()) == set(ZT_VALUES)
write_csv(design, MOD, "pseudobulk_design.csv", "Pseudobulk design: cells and library size per group × sample",
          required=["tissue", "level", "group", "sample_id", "zt", "n_cells", "lib_size", "status"])
write_csv(incl, MOD, "groups_included.csv", "Inclusion status of each tissue × level × group for rhythm analyses",
          required=["tissue", "level", "group", "status"])
print(incl.groupby(["level", "tissue", "status"]).size().to_string())
assert pd.util.hash_pandas_object(adata.obs[LABEL_COLS].astype(str), index=True).sum() == label_hash
log("done")
