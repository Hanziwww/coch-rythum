"""Cross-tabulate the circadian label (day_time) against sample-level metadata,
and check whether .X holds raw counts. Read-only; backed mode."""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)

H5AD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "data/coca_main_atlas_with_obs.h5ad")

adata = ad.read_h5ad(H5AD, backed="r")
obs = adata.obs.copy()
obs["day_time_str"] = obs["day_time"].map(lambda v: "NA" if pd.isna(v) else f"{v:g}")
for col in ["batch_id", "condition", "treat_time", "tissue", "platform", "cell_class", "cell_type"]:
    obs[col] = obs[col].astype(object).where(obs[col].notna(), "NA").astype(str)

meta_cols = ["batch_id", "condition", "treat_time", "tissue", "platform", "day_time_str"]

print("=== per-batch sample metadata (one row per batch) ===")
per_batch = (obs[meta_cols]
             .groupby("batch_id", observed=True)
             .agg(lambda s: "|".join(sorted(s.unique()))))
per_batch["n_cells"] = obs.groupby("batch_id", observed=True).size()
print(per_batch.sort_index().to_string())

print("\n=== batches with mixed day_time (labelled and unlabelled cells) ===")
mixed = per_batch[per_batch["day_time_str"].str.contains(r"\|")]
print(mixed.to_string() if len(mixed) else "none")

for col in ["condition", "treat_time", "tissue", "platform"]:
    print(f"\n=== day_time x {col} ===")
    print(pd.crosstab(obs["day_time_str"], obs[col], margins=True).to_string())

print("\n=== day_time x cell_class ===")
print(pd.crosstab(obs["cell_class"], obs["day_time_str"], margins=True).to_string())

print("\n=== day_time x cell_type ===")
print(pd.crosstab(obs["cell_type"], obs["day_time_str"], margins=True).to_string())

print("\n=== .X content check (first 2000 cells) ===")
x = adata.X[:2000]
x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
nz = x[x > 0]
print("min/max:", x.min(), x.max())
print("fraction non-integer among nonzeros:", np.mean(np.abs(nz - np.round(nz)) > 1e-6))
print("row sums (first 10):", np.round(x.sum(1)[:10], 2))
print("obs total_counts (first 10):", obs["total_counts"].values[:10])
