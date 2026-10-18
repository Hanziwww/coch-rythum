"""Build stand-in inputs to dry-run the Ma 2025 pipeline (300 → 301 → 302 → 310 → 311 → 312, 314) before the user's
matrices arrive. Nothing here is analysed as Ma 2025 data; outputs of the dry run are written under key 'ma2025dry'
and deleted afterwards.
- scRNA: ten 10x 'filtered_feature_bc_matrix' folders named as in data/CRA019820.xlsx, each a random subset
  (2,000 cells) of a GSE346691 sample.
- bulk: GSE107086 (control / noise × day / night, n = 3; sham surgery) renamed to CMD/CMN/NMD/NMN 1–3, written as
  CRA019827_counts_dryrun.csv. Its true sampling times (~ZT5.5 / ZT17.5, round-2 pilot) test the phase inference.
Output folder: cache/ma2025_dryrun/ (MA_ROOT)
"""
import gzip
import io
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.io import CACHE, EXTERNAL, log  # noqa: E402

root = CACHE / "ma2025_dryrun"
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
q = ad.read_h5ad(CACHE / "r3_query/gse346691_qc.h5ad")
rng = np.random.default_rng(1)
plan = {"C1_ctrl": "WT_CON2", "C2_ctrl": "WT_CON3", "C3_ctrl": "WT_CON2", "D1_Ne3d": "WT_110dB", "D2_Ne3d": "WT",
        "D3_Ne3d": "WT_NE3", "D4_Ne3d": "WT_110dB", "N1_Ne3d": "Spp1KO_110dB", "N2_Ne3d": "Ccr2_NE", "N3_Ne3d": "WT"}
for name, src in plan.items():
    idx = np.where(q.obs["sample"] == src)[0]
    idx = rng.choice(idx, min(2000, len(idx)), replace=False)
    x = q[idx]
    d = root / name / "filtered_feature_bc_matrix"
    d.mkdir(parents=True)
    with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
        buf = io.BytesIO()
        scipy.io.mmwrite(buf, x.X.T.tocoo().astype(np.int64))
        fh.write(buf.getvalue())
    gid = x.var["gene_ids"].astype(str).to_numpy() if "gene_ids" in x.var else x.var_names.to_numpy()
    pd.DataFrame({"id": gid, "symbol": x.var_names, "type": "Gene Expression"}).to_csv(d / "features.tsv.gz", sep="\t", header=False, index=False)
    pd.Series([b.split(":")[-1] for b in x.obs_names]).to_csv(d / "barcodes.tsv.gz", sep="\t", header=False, index=False)
txt = gzip.open(EXTERNAL / "geo/GSE107086/GSE107086_raw_counts.csv.gz", "rt", newline="").read().replace("\r", "\n")
c = pd.read_csv(io.StringIO(txt)).dropna(how="all").drop_duplicates("geneName").set_index("geneName")
ren = {}
for pre, new in [("ShDNShADX", "CMD"), ("ShNNShADX", "CMN"), ("DNShADX", "NMD"), ("NNShADX", "NMN")]:
    for i, s in enumerate("abc", 1):
        ren[f"{pre}{s}"] = f"{new}{i}"
c[list(ren)].rename(columns=ren).to_csv(root / "CRA019827_counts_dryrun.csv")
log(f"dry-run inputs written to {root}")
