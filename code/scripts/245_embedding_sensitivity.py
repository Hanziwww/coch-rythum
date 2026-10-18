"""Round 2, WP5 — are embedding-based results tied to the label-supervised scANVI embeddings?

(a) Milo (53) re-run for BM and cochlea on the unsupervised scVI embeddings of the same families (C__scvi_none_nl30_s0):
    this script writes the latent CSVs to cache/r2_embedding/ and the R step is run with LATENT_TEMPLATE / OUT_SUFFIX.
(b) BM-likeness of cochlear immune cells (66) recomputed on the joint unsupervised scVI embedding
    (C__scvi_platform_nl50_s0): fraction of BM cells among 30 nearest neighbours, normalised by the cell type's global
    BM fraction; per-sample means; cosinor with 60-class permutation p — compared with the scANVI-based result.
Outputs: cache/r2_embedding/latent_<tissue>_scvi.csv (internal); results/r2_audit/bm_likeness_scvi_rhythm.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, cache_dir, log, write_csv  # noqa: E402

out = cache_dir("r2_embedding")
for tissue in ["bm", "cochlea"]:
    ref = pd.read_csv(RESULTS / f"20_dr/embedding_{tissue}_latent.csv", usecols=["cell_id"])
    Z = np.load(CACHE / "20_dr" / tissue / "emb" / "C__scvi_none_nl30_s0.npy")
    assert len(Z) == len(ref)
    pd.concat([ref, pd.DataFrame(Z, columns=[f"scvi_{i}" for i in range(Z.shape[1])])], axis=1).to_csv(
        out / f"latent_{tissue}_scvi.csv", index=False)
    log(f"{tissue}: scVI latent {Z.shape} written")

obs = pd.read_parquet(CACHE / "20_dr/joint/obs.parquet")
Z = np.load(CACHE / "20_dr/joint/emb/C__scvi_platform_nl50_s0.npy")
assert len(Z) == len(obs)
nn = NearestNeighbors(n_neighbors=31, n_jobs=6).fit(Z)
_, ind = nn.kneighbors(Z)
is_bm = (obs["tissue_code"].astype(str) == "bm").to_numpy()
frac = is_bm[ind[:, 1:]].mean(1)
df = obs[["tissue_code", "sample_id", "zt", "cell_type"]].astype({"tissue_code": str, "sample_id": str, "cell_type": str}).copy()
df["zt"] = pd.to_numeric(df["zt"])
df["frac_bm_neighbours"] = frac
bm_frac = df.groupby("cell_type")["tissue_code"].apply(lambda s: (s == "bm").mean())
df["bm_likeness"] = df["frac_bm_neighbours"] / df["cell_type"].map(bm_frac).astype(float).clip(lower=1e-3)
IMM = ["Granulocyte-Monocyte Progenitor", "Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)",
       "Neutrophil (Activated)", "Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage", "Pro-B Cell (Cycling)",
       "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)", "T Cell (CD3+)", "NK Cell", "pDC",
       "Mast Cell"]
co = df[(df["tissue_code"] == "cochlea") & df["cell_type"].isin(IMM)]
bs = co.groupby(["cell_type", "sample_id", "zt"])["bm_likeness"].agg(["size", "mean"]).reset_index()
old = pd.read_csv(RESULTS / "66_state_continua/bm_likeness_rhythm.csv").set_index("cell_type")
rows = []
for c, g in bs.groupby("cell_type"):
    g = g.sort_values("zt")
    if len(g) < 6 or g["size"].min() < 10:
        continue
    Y = g[["mean"]].to_numpy()
    f = cosinor.fit(Y, g["zt"].to_numpy())
    _, _, pp, _ = cosinor.permutation_test(Y, g["zt"].to_numpy())
    rows.append({"cell_type": c, "embedding": "joint scVI (platform), unsupervised", "mean_bm_likeness": float(Y.mean()),
                 "peak_zt": float(f["peak"][0]), "p": float(f["p"][0]), "p_perm": float(pp[0]),
                 "scanvi_peak_zt": old["peak_zt"].get(c, np.nan), "scanvi_p": old["p"].get(c, np.nan)})
write_csv(pd.DataFrame(rows), "r2_audit", "bm_likeness_scvi_rhythm.csv",
          "BM-likeness rhythm of cochlear immune cells on the unsupervised joint scVI embedding vs scANVI",
          required=["cell_type", "peak_zt", "p", "scanvi_p"])
log("done")
