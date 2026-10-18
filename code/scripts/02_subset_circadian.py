"""Step 1: keep only the circadian time-course samples (cells with a day_time label).

Input : data/coca_main_atlas_with_obs.h5ad  (full atlas, raw counts in .X)
Output: data/coca_circadian.h5ad            (12 samples: 6 time points x {cochlea, bone marrow})

Cell-type annotations (cell_type, cell_class, cell_type_abbr) are carried over
unchanged and treated as the gold standard.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
SRC = ROOT / "data" / "coca_main_atlas_with_obs.h5ad"
DST = ROOT / "data" / "coca_circadian.h5ad"
ANNOT_COLS = ["cell_type", "cell_class", "cell_type_abbr"]
EXPECTED_TIMES = {3.0, 7.0, 11.0, 15.0, 19.0, 23.0}

adata = ad.read_h5ad(SRC)
print("full atlas:", adata.shape)
assert adata.obs_names.is_unique

keep = adata.obs["day_time"].notna().to_numpy()
sub = adata[keep].copy()
del adata

# Drop category levels that no longer occur (e.g. other conditions/tissues).
for col in sub.obs.select_dtypes("category").columns:
    sub.obs[col] = sub.obs[col].cat.remove_unused_categories()

# --- sanity checks -----------------------------------------------------------
obs = sub.obs
assert set(obs["day_time"].unique()) == EXPECTED_TIMES, obs["day_time"].unique()
assert (obs["condition"] == "C").all()
# every sample carries exactly one time point and one tissue
per_batch = obs.groupby("batch_id", observed=True)[["day_time", "tissue", "platform"]].nunique()
assert (per_batch == 1).all().all(), per_batch
# each (time point, tissue) combination is one sample
design = obs.groupby(["day_time", "tissue"], observed=True)["batch_id"].nunique()
assert (design == 1).all() and len(design) == 12, design

# annotations must be identical to the source atlas for the retained cells
src_obs = ad.read_h5ad(SRC, backed="r").obs.loc[obs.index, ANNOT_COLS]
for col in ANNOT_COLS:
    assert (src_obs[col].astype(str).to_numpy() == obs[col].astype(str).to_numpy()).all(), col

x = sub.X[:5000]
x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
assert np.allclose(x, np.round(x)), ".X is expected to hold raw counts"

# --- summary -----------------------------------------------------------------
print("circadian subset:", sub.shape)
summary = (obs.groupby("batch_id", observed=True)
           .agg(day_time=("day_time", "first"), tissue=("tissue", "first"),
                platform=("platform", "first"), n_cells=("day_time", "size"))
           .sort_values(["tissue", "day_time"]))
print(summary.to_string())
print("\ncells per tissue x day_time:")
print(pd.crosstab(obs["tissue"], obs["day_time"], margins=True).to_string())
print("\ncell types retained:", obs["cell_type"].nunique())

sub.write_h5ad(DST, compression="gzip")
print("\nwritten:", DST)
