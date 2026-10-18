"""Export raw counts + cell metadata (MatrixMarket) for 67_multinichenet.R (internal input, not a deliverable)."""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import H5AD, cache_dir, log, sample_table  # noqa: E402

a = ad.read_h5ad(H5AD)
st = sample_table(a.obs).set_index("sample_id")
d = cache_dir("67_multinichenet")
meta = pd.DataFrame({"cell_id": a.obs_names, "sample_id": a.obs["batch_id"].astype(str).values,
                     "cell_type": a.obs["cell_type"].astype(str).values})
meta["tissue"] = meta["sample_id"].map(st["tissue"])
meta["zt"] = meta["sample_id"].map(st["zt"])
meta.to_csv(d / "cells.csv", index=False)
pd.DataFrame({"gene": a.var_names}).to_csv(d / "genes.csv", index=False)
scipy.io.mmwrite(d / "counts.mtx", a.X.tocsr().astype("int32"))
log(f"exported {a.n_obs} cells × {a.n_vars} genes")
