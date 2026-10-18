"""Module 1, Stages B/C — train integration/embedding methods on the Stage-A feature set of a family.

Usage:
  python scripts/21_dr_train.py <family> B <hvg_config>
  python scripts/21_dr_train.py <family> C <hvg_config> <method> [<method> ...]
Stage B: every method at defaults (n_latent/n_pcs 30, seed 0).
Stage C: for the given methods, n_latent {20, 30, 50} at seed 0, then seeds 1, 2 at the best n_latent from the
         Stage-C metrics (pass the best value with env BEST_NLAT to run the seed replicates only).
Methods: pca, harmony_sample, harmony_platform, scvi_none, scvi_sample, scvi_platform, scanvi_*, sysvi,
         scanorama_sample, scanorama_platform, bbknn_platform (diffusion map of the BBKNN graph).
scANVI masks the gold-standard labels of the fixed evaluation subsample so its bio metrics are not circular.
Embeddings are written to cache/20_dr/<family>/emb/<stage>__<method>_nl<k>_s<seed>.npy (+ .json with params).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import dr  # noqa: E402
from lib.io import cache_dir, log  # noqa: E402

FAM, STAGE, HVG = sys.argv[1], sys.argv[2], sys.argv[3]
root = cache_dir(f"20_dr/{FAM}")
genes = (root / "hvg" / f"{HVG}.txt").read_text().split()
tissue = None if FAM == "joint" else FAM
a_full = dr.load(tissue)
lib = np.asarray(a_full.X.sum(1)).ravel()
a = a_full[:, genes].copy()
a.obs["lib"] = lib
del a_full
eval_idx = np.load(root / "eval_idx.npy")
norm = "lognorm" if "seurat_v3" in HVG else "pearson"
Z_pca50 = np.load(root / "emb" / f"{HVG}.npy")
n_cells = a.n_obs
default_epochs = int(min(round(20000 / n_cells * 400), 400))


def save(name, Z, params, t0):
    np.save(root / "emb" / f"{name}.npy", Z.astype(np.float32))
    params.update({"hvg": HVG, "seconds": round(time.time() - t0, 1), "dims": int(Z.shape[1])})
    (root / "emb" / f"{name}.json").write_text(json.dumps(params))
    log(f"  saved {name} {Z.shape} in {params['seconds']}s")


def a_full_like():
    # lognorm with full-transcriptome library sizes: rebuild from counts and stored lib
    import scipy.sparse as sp
    X = sp.diags(1e4 / np.maximum(a.obs["lib"].to_numpy(), 1)).astype(np.float32) @ a.X.tocsr()
    X.data = np.log1p(X.data)
    return X


def run(method, n_lat=30, seed=0, stage=STAGE):
    name = f"{stage}__{method}_nl{n_lat}_s{seed}"
    if (root / "emb" / f"{name}.npy").exists():
        log(f"  exists {name}")
        return
    t0 = time.time()
    params = {"method": method, "n_latent": n_lat, "seed": seed}
    if method == "pca":
        save(name, Z_pca50[:, :n_lat], params, t0)
        return
    if method.startswith("harmony_"):
        import harmonypy
        key = {"harmony_sample": "sample_id", "harmony_platform": "tissue_code"}[method]
        ho = harmonypy.run_harmony(Z_pca50[:, :n_lat], a.obs[[key]], [key], random_state=seed, verbose=False)
        Z = np.asarray(ho.Z_corr)
        save(name, Z.T if Z.shape[0] == n_lat else Z, params, t0)
        return
    if method.startswith("scanorama_"):
        key = {"scanorama_sample": "sample_id", "scanorama_platform": "tissue_code"}[method]
        order = np.argsort(a.obs[key].to_numpy(), kind="stable")
        tmp = ad.AnnData(np.zeros((n_cells, 1)), obs=a.obs[[key]].iloc[order].copy())
        tmp.obsm["X_pca"] = Z_pca50[order, :50]
        sc.external.pp.scanorama_integrate(tmp, key, basis="X_pca", adjusted_basis="X_sc", verbose=0)
        Z = np.empty((n_cells, tmp.obsm["X_sc"].shape[1]), dtype=np.float32)
        Z[order] = tmp.obsm["X_sc"]
        save(name, Z[:, :n_lat], params, t0)
        return
    if method == "bbknn_platform":
        import bbknn
        tmp = ad.AnnData(np.zeros((n_cells, 1)), obs=a.obs[["tissue_code"]].copy())
        tmp.obsm["X_pca"] = Z_pca50[:, :n_lat]
        bbknn.bbknn(tmp, batch_key="tissue_code", use_rep="X_pca", neighbors_within_batch=7)
        sc.tl.diffmap(tmp, n_comps=n_lat + 1)
        save(name, tmp.obsm["X_diffmap"][:, 1:], params, t0)
        return
    import scvi
    import torch
    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = seed
    scvi.settings.dl_num_workers = 0
    train_kw = dict(max_epochs=int(default_epochs * 1.5), batch_size=1024, early_stopping=True,
                    early_stopping_patience=10, accelerator="gpu", devices=1, enable_progress_bar=False)
    if method.startswith(("scvi_", "scanvi_")):
        batch = {"none": None, "sample": "sample_id", "platform": "tissue_code"}[method.split("_", 1)[1]]
        b = a.copy()
        scvi.model.SCVI.setup_anndata(b, batch_key=batch)
        vae = scvi.model.SCVI(b, n_latent=n_lat, n_layers=2, n_hidden=128, gene_likelihood="nb")
        vae.train(**train_kw)
        params["epochs"] = int(vae.history["elbo_train"].shape[0])
        if method.startswith("scvi_"):
            save(name, vae.get_latent_representation(), params, t0)
            return
        lab = b.obs["cell_type"].astype(str).to_numpy().copy()
        lab[eval_idx] = "Unknown"
        b.obs["label_masked"] = pd.Categorical(lab)                 # vae.adata is b
        lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key="label_masked")
        lvae.train(max_epochs=20, n_samples_per_label=100, batch_size=1024, accelerator="gpu", devices=1,
                   enable_progress_bar=False)
        save(name, lvae.get_latent_representation(), params, t0)
        return
    if method == "sysvi":
        from scvi.external import SysVI
        b = ad.AnnData(a_full_like(), obs=a.obs[["tissue_code", "sample_id"]].copy())
        b.var_names = genes
        SysVI.setup_anndata(b, batch_key="tissue_code", categorical_covariate_keys=["sample_id"])
        w = float(os.environ.get("SYSVI_CYC", "5"))
        m = SysVI(b, prior="vamp", n_prior_components=5, n_latent=n_lat)
        m.train(max_epochs=int(default_epochs * 1.5), batch_size=1024, early_stopping=True,
                early_stopping_patience=10, accelerator="gpu", devices=1, enable_progress_bar=False,
                plan_kwargs={"kl_weight": 1.0, "reconstruction_weight": 1.0, "z_distance_cycle_weight": w})
        params["z_distance_cycle_weight"] = w
        save(name + (f"_cyc{int(w)}" if "SYSVI_CYC" in os.environ else ""), m.get_latent_representation(),
             params, t0)
        return
    raise ValueError(method)


if STAGE == "B":
    if FAM == "joint":
        methods = ["pca", "harmony_platform", "harmony_sample", "scvi_platform", "scvi_sample", "scanvi_platform",
                   "sysvi", "scanorama_platform", "bbknn_platform"]
    else:
        methods = ["pca", "harmony_sample", "scvi_none", "scvi_sample", "scanvi_none", "scanorama_sample"]
    log(f"{FAM} Stage B on {HVG}: {methods} (default epochs {default_epochs})")
    for m in methods:
        try:
            run(m)
        except Exception as e:  # keep going; failures are reported
            log(f"  FAILED {m}: {type(e).__name__}: {e}")
else:
    methods = sys.argv[4:]
    best = os.environ.get("BEST_NLAT")
    for m in methods:
        if best:
            for s in (1, 2):
                run(m, int(best), s)
        else:
            for nl in (20, 30, 50):
                run(m, nl, 0)
log("done")
