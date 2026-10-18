"""Inspect the cochlear atlas: obs columns, sample-level metadata, and candidate
circadian/time-of-day labels. Read-only; loads the h5ad in backed mode."""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)

H5AD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "data/coca_main_atlas_with_obs.h5ad")

adata = ad.read_h5ad(H5AD, backed="r")
print(adata)
print("\nX dtype:", adata.X.dtype if hasattr(adata.X, "dtype") else type(adata.X))
print("layers:", list(adata.layers.keys()))
print("raw:", adata.raw is not None)

obs = adata.obs
print("\n=== obs dtypes / n_unique ===")
summary = pd.DataFrame({
    "dtype": obs.dtypes.astype(str),
    "n_unique": obs.nunique(dropna=False),
    "n_na": obs.isna().sum(),
})
print(summary)

print("\n=== value counts for low-cardinality columns (<=80 levels) ===")
for col in obs.columns:
    n = obs[col].nunique(dropna=False)
    if n <= 80:
        print(f"\n--- {col} ({n}) ---")
        print(obs[col].value_counts(dropna=False).to_string())

print("\n=== columns whose name or values suggest time / circadian labels ===")
pat = r"(?i)circ|rhythm|zt|ct\d|time|clock|hour|light|dark|day|night|\bZT"
for col in obs.columns:
    hit_name = pd.Series([col]).str.contains(pat).iloc[0]
    vals = obs[col].astype(str).unique()[:200]
    hit_vals = pd.Series(vals).str.contains(pat).any()
    if hit_name or hit_vals:
        print(f"  {col}: name_hit={hit_name}, value_hit={hit_vals}")

print("\n=== uns keys ===")
print(list(adata.uns.keys()))
