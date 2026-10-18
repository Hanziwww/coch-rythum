"""Round 3 — mapping external single-cell data onto the atlas gold-standard labels (labels are never changed).

Two independent transfers, then a consensus:
  (i)  scANVI reference (cochlea; genes = the 5,000 HVGs of the selected cochlear embedding, A__seurat_v3_5000_ex0;
       scVI 30 latent / 2 layers / 128 hidden / NB as in 21_dr_train.py, but scArches-ready: sample as batch, layer
       norm, no batch norm, encoded covariates) + scArches query surgery (new batches = external samples);
  (ii) PCA-kNN as in 223_skull_marrow_reference.py (log-normalised, 3,000 seurat HVGs of the reference, PCA 50 fitted
       on the reference, distance-weighted kNN k = 15).
Consensus label = scANVI label; confident = both methods agree, scANVI probability >= 0.7 and kNN vote >= 0.5.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from lib.io import CACHE, H5AD, log

REF_DIR = CACHE / "r3_refmap"
HVG_FILE = CACHE / "20_dr/cochlea/hvg/A__seurat_v3_5000_ex0.txt"


def load_reference(tissue="cochlea"):
    a = ad.read_h5ad(H5AD)
    a = a[(a.obs["tissue"].astype(str) == tissue).to_numpy()].copy()
    a.obs["sample_id"] = a.obs["batch_id"].astype(str)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    return a


def train_reference(seed=0, force=False):
    import scvi
    import torch
    out = REF_DIR / "cochlea_scanvi"
    if (out / "model.pt").exists() and not force:
        return out
    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = seed
    scvi.settings.dl_num_workers = 0
    ref = load_reference()
    genes = [g for g in HVG_FILE.read_text().split() if g in ref.var_names]
    ref = ref[:, genes].copy()
    ref.obs = ref.obs[["sample_id", "cell_type"]].copy()
    scvi.model.SCVI.setup_anndata(ref, batch_key="sample_id")
    vae = scvi.model.SCVI(ref, n_latent=30, n_layers=2, n_hidden=128, gene_likelihood="nb", use_layer_norm="both",
                          use_batch_norm="none", encode_covariates=True, dropout_rate=0.2)
    vae.train(max_epochs=200, batch_size=1024, early_stopping=True, early_stopping_patience=10, accelerator="gpu",
              devices=1, enable_progress_bar=False)
    lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key="cell_type")
    lvae.train(max_epochs=20, n_samples_per_label=100, batch_size=1024, accelerator="gpu", devices=1, enable_progress_bar=False)
    out.mkdir(parents=True, exist_ok=True)
    lvae.save(out, overwrite=True, save_anndata=False)
    acc = float((lvae.predict() == ref.obs["cell_type"].to_numpy()).mean())
    log(f"reference trained: {ref.n_obs} cells × {ref.n_vars} genes; self-prediction accuracy {acc:.3f}")
    return out


def _lognorm(x):
    x = x.copy()
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    return x


def map_query(q, batch_key="sample", seed=0, max_epochs=100):
    """q: AnnData of raw counts, var_names = gene symbols, obs[batch_key] = external sample. Returns per-cell table."""
    import scvi
    scvi.settings.seed = seed
    ref_dir = train_reference()
    # (i) scArches
    qq = q.copy()
    qq.obs = pd.DataFrame({"sample_id": "ext_" + q.obs[batch_key].astype(str).to_numpy(), "cell_type": "Unknown"},
                          index=q.obs_names)
    scvi.model.SCANVI.prepare_query_anndata(qq, str(ref_dir))
    qm = scvi.model.SCANVI.load_query_data(qq, str(ref_dir))
    qm.train(max_epochs=max_epochs, batch_size=1024, plan_kwargs={"weight_decay": 0.0}, accelerator="gpu", devices=1,
             enable_progress_bar=False)
    soft = qm.predict(soft=True)
    res = pd.DataFrame({"scanvi_label": soft.idxmax(1).to_numpy(), "scanvi_prob": soft.max(1).to_numpy()}, index=q.obs_names)
    Z = qm.get_latent_representation()
    # (ii) PCA-kNN
    ref = load_reference()
    genes = ref.var_names.intersection(q.var_names)
    r2, q2 = _lognorm(ref[:, genes]), _lognorm(q[:, genes])
    sc.pp.highly_variable_genes(r2, n_top_genes=3000, flavor="seurat")
    hv = r2.var_names[r2.var["highly_variable"]]
    Xr = r2[:, hv].X.toarray()
    Xq = q2[:, hv].X.toarray()
    mu = Xr.mean(0)
    pca = PCA(n_components=50, random_state=seed).fit(Xr - mu)
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(pca.transform(Xr - mu), r2.obs["cell_type"].to_numpy())
    proba = knn.predict_proba(pca.transform(Xq - mu))
    res["knn_label"] = knn.classes_[proba.argmax(1)]
    res["knn_vote"] = proba.max(1)
    res["label"] = res["scanvi_label"]
    res["agree"] = res["scanvi_label"] == res["knn_label"]
    res["confident"] = res["agree"] & (res["scanvi_prob"] >= 0.7) & (res["knn_vote"] >= 0.5)
    res["n_shared_genes"] = len(genes)
    log(f"mapped {q.n_obs} cells; agreement {res['agree'].mean():.2f}; confident {res['confident'].mean():.2f}")
    return res, Z
