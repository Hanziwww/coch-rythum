"""Neutrophil-lineage filter for external single-cell references (round 6).

Sorted or whole-tissue neutrophil datasets carry lymphocytes, NK cells and monocytes; signatures built from them must
use the neutrophil lineage only. Input: AnnData with log CP10k in X and raw counts in layers["counts"].
Leiden clusters (HVG 2,000, PCA 30, kNN 15, resolution 1.0) are kept when the mean neutrophil marker score exceeds
both the lymphoid / NK and the monocyte scores and is positive.
"""
import numpy as np
import pandas as pd
import scanpy as sc

NEU_MARK = ["S100a8", "S100a9", "Ngp", "Camp", "Ltf", "Lcn2", "Retnlg", "Cxcr2", "Elane", "Mpo"]
LYM_MARK = ["Cd3e", "Cd3g", "Lck", "Cd79a", "Cd79b", "Ms4a1", "Nkg7", "Gzma"]
MONO_MARK = ["Csf1r", "Cx3cr1", "Ly6c2", "Ccr2", "Ms4a6c", "H2-Aa"]


def neutrophil_lineage(X, seed=0):
    """Return (filtered copy, note)."""
    sc.pp.highly_variable_genes(X, n_top_genes=2000, flavor="seurat_v3", layer="counts")
    sc.pp.pca(X, n_comps=30, mask_var="highly_variable")
    sc.pp.neighbors(X, n_neighbors=15, random_state=seed)
    sc.tl.leiden(X, resolution=1.0, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    for nm_, gl in {"neu": NEU_MARK, "lym": LYM_MARK, "mono": MONO_MARK}.items():
        sc.tl.score_genes(X, [g for g in gl if g in X.var_names], score_name=f"s_{nm_}", random_state=seed)
    cl = X.obs.groupby("leiden", observed=True)[["s_neu", "s_lym", "s_mono"]].mean()
    keep = cl.index[(cl["s_neu"] > cl["s_lym"]) & (cl["s_neu"] > cl["s_mono"]) & (cl["s_neu"] > 0)]
    n0 = X.n_obs
    Y = X[X.obs["leiden"].isin(keep).to_numpy()].copy()
    return Y, f"{Y.n_obs} of {n0} cells in {len(keep)} neutrophil clusters"


def contrast_signature(X, hi, lo, n=50, min_det=0.10, gene_ok=None):
    """Cell-level contrast signature (robust to minority contaminants that dominate summed pseudobulk).

    X: log CP10k in X; hi / lo: boolean cell masks. Difference of mean log CP10k (hi − lo); up genes must be detected in
    >= min_det of hi cells, down genes in >= min_det of lo cells. Returns (up, down) Series of the top n each way.
    """
    def mean_det(m):
        S = X.X[np.asarray(m)]
        return np.asarray(S.mean(0)).ravel(), np.asarray((S > 0).mean(0)).ravel()

    mh, dh = mean_det(hi)
    ml, dl = mean_det(lo)
    d = pd.Series(mh - ml, index=X.var_names)
    ok = np.array([gene_ok(g) for g in d.index]) if gene_ok else np.ones(len(d), bool)
    up = d[ok & (dh >= min_det) & (d.to_numpy() > 0)].sort_values(ascending=False).head(n)
    dn = d[ok & (dl >= min_det) & (d.to_numpy() < 0)].sort_values().head(n)
    return up, dn
