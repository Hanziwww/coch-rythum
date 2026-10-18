"""Module 1, Stage A — feature selection / normalisation screen with PCA only.

Families: cochlea, bm (per-tissue, HVGs batch-aware over samples) and joint (HVGs batch-aware over tissue/platform).
Configs: HVG flavour {seurat_v3 → log-CP10k+scale PCA, pearson → analytic Pearson residual PCA} × n {3000, 5000}
× exclusion of mt/ribo/Hb/stress genes {no, yes}. Also fixes the evaluation subsamples (label-stratified) used by
every later metric computation and by scANVI label masking.

Outputs: cache/20_dr/{family}/emb/A__*.npy, cache/20_dr/{family}/hvg/*.txt, cache/20_dr/{family}/eval_idx.npy,
         results/20_dr/dr_hvg_sets.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import dr  # noqa: E402
from lib.io import cache_dir, log, write_csv  # noqa: E402

FAMILIES = {"cochlea": dict(tissue="cochlea", hvg_batch="sample_id", n_eval=30000),
            "bm": dict(tissue="bm", hvg_batch="sample_id", n_eval=20000),
            "joint": dict(tissue=None, hvg_batch="tissue_code", n_eval=40000)}
CONFIGS = [(fl, n, ex) for fl in ["seurat_v3", "pearson"] for n in [3000, 5000] for ex in [False, True]]

hvg_rows = []
full = dr.load()
for fam, spec in FAMILIES.items():
    a = full if spec["tissue"] is None else full[full.obs["tissue_code"] == spec["tissue"]].copy()
    root = cache_dir(f"20_dr/{fam}")
    (root / "emb").mkdir(exist_ok=True)
    (root / "hvg").mkdir(exist_ok=True)
    a.obs[["sample_id", "tissue_code", "zt", "cell_type", "cell_class"]].to_parquet(root / "obs.parquet")
    eval_idx = dr.stratified_subsample(a.obs, spec["n_eval"], seed=0)
    np.save(root / "eval_idx.npy", eval_idx)
    log(f"{fam}: {a.n_obs} cells, eval subsample {len(eval_idx)}")
    for fl, n, ex in CONFIGS:
        name = f"A__{fl}_{n}_ex{int(ex)}"
        t0 = time.time()
        genes = dr.select_hvg(a, fl, n, spec["hvg_batch"], ex)
        (root / "hvg" / f"{name}.txt").write_text("\n".join(genes))
        Z = dr.pca_embedding(a, genes, norm="lognorm" if fl == "seurat_v3" else "pearson", n_pcs=50)
        np.save(root / "emb" / f"{name}.npy", Z)
        hvg_rows += [{"family": fam, "config": name, "gene": g, "rank": i + 1} for i, g in enumerate(genes)]
        log(f"  {name}: {len(genes)} genes, PCA {Z.shape} in {time.time() - t0:.0f}s")
    if spec["tissue"] is None:
        np.save(root / "emb" / "ref__orig_umap.npy", a.obsm["X_umap"].astype(np.float32))

write_csv(pd.DataFrame(hvg_rows), "20_dr", "dr_hvg_sets.csv", "Stage A HVG sets per embedding family and config",
          required=["family", "config", "gene", "rank"])
log("done")
