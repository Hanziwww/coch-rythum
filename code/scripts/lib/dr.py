"""Dimensionality-reduction helpers for Module 1 (feature selection, embeddings, evaluation subsets)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad

from .io import H5AD, TISSUE_FROM_OBS, sample_table

STRESS = ["Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36",
          "Hspa1a", "Hspa1b", "Hspa8", "Hsph1", "Dnajb1", "Btg2", "Klf2", "Klf4", "Klf6", "Socs3", "Ppp1r15a",
          "Cebpd", "Ccn1", "Cyr61", "Gadd45b", "Gadd45g", "Ier5"]


def load(tissue: str | None = None) -> ad.AnnData:
    """Raw-count AnnData with sample_id, tissue_code, zt, clock_time added to obs (optionally one tissue)."""
    a = ad.read_h5ad(H5AD)
    st = sample_table(a.obs).set_index("sample_id")
    a.obs["sample_id"] = a.obs["batch_id"].astype(str)
    a.obs["tissue_code"] = a.obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
    a.obs["zt"] = a.obs["sample_id"].map(st["zt"]).astype(float)
    a.obs["clock_time"] = a.obs["sample_id"].map(st["clock_time"]).astype(float)
    if tissue is not None:
        a = a[a.obs["tissue_code"] == tissue].copy()
    a.X = a.X.tocsr().astype(np.float32)
    return a


def exclusion_mask(genes) -> np.ndarray:
    g = pd.Index(genes)
    return np.asarray(g.str.startswith("mt-") | g.str.match(r"^Rp[sl]\d") | g.str.match(r"^Hb[ab]-")
                      | g.isin(STRESS))


def select_hvg(a: ad.AnnData, flavor: str, n: int, batch_key: str | None, exclude: bool) -> list[str]:
    """Top-n HVGs from raw counts; 'seurat_v3' or 'pearson' (analytic Pearson residuals)."""
    extra = 1500 if exclude else 0
    if flavor == "seurat_v3":
        df = sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n + extra, batch_key=batch_key,
                                         inplace=False, span=0.3)
    elif flavor == "pearson":
        df = sc.experimental.pp.highly_variable_genes(a, flavor="pearson_residuals", n_top_genes=n + extra,
                                                      batch_key=batch_key, inplace=False, theta=100)
    else:
        raise ValueError(flavor)
    df = df.copy()
    df.index = a.var_names
    df = df[df["highly_variable"]]
    if exclude:
        df = df[~exclusion_mask(df.index)]
    sort_cols = [c for c in ["highly_variable_nbatches", "highly_variable_rank"] if c in df.columns]
    asc = [False, True][-len(sort_cols):] if len(sort_cols) == 2 else [True]
    df = df.sort_values(sort_cols, ascending=asc)
    return list(df.index[:n])


def lognorm(a: ad.AnnData, genes: list[str]) -> sp.csr_matrix:
    """log1p(CP10k) using full-transcriptome library size, restricted to genes."""
    lib = np.asarray(a.X.sum(1)).ravel()
    X = a[:, genes].X.tocsr().astype(np.float32)
    X = sp.diags(1e4 / np.maximum(lib, 1)).astype(np.float32) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def pca_embedding(a: ad.AnnData, genes: list[str], norm: str = "lognorm", n_pcs: int = 50,
                  seed: int = 0) -> np.ndarray:
    if norm == "lognorm":
        X = lognorm(a, genes)
        tmp = ad.AnnData(X)
        sc.pp.scale(tmp, max_value=10)
    elif norm == "pearson":
        tmp = ad.AnnData(a[:, genes].X.copy())
        sc.experimental.pp.normalize_pearson_residuals(tmp, theta=100)
    else:
        raise ValueError(norm)
    sc.pp.pca(tmp, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return tmp.obsm["X_pca"].astype(np.float32)


def stratified_subsample(obs: pd.DataFrame, n_target: int, seed: int = 0,
                         keys=("tissue_code", "cell_type", "sample_id")) -> np.ndarray:
    """Indices of a label-stratified subsample: equal cap per (tissue, cell type, sample) cell, rare cells kept."""
    rng = np.random.default_rng(seed)
    groups = obs.groupby(list(keys), observed=True).indices
    sizes = np.array([len(v) for v in groups.values()])
    lo, hi = 1, int(sizes.max())
    while lo < hi:                                   # largest cap with total <= n_target
        mid = (lo + hi + 1) // 2
        if np.minimum(sizes, mid).sum() <= n_target:
            lo = mid
        else:
            hi = mid - 1
    idx = [rng.choice(v, size=min(len(v), lo), replace=False) for v in groups.values()]
    return np.sort(np.concatenate(idx))
