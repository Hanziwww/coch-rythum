"""Round 9 (D60 / D61) — status-N extractions from the h5ad for Fig. 1 and ED1 / ED2 (results/r9_figures).

Descriptive only: no inference, no re-annotation (cell_type / cell_class are the gold standard and are only joined).
Expression = log1p(counts / total × 1e4) per cell; pseudobulk = summed raw counts per cell type.

Outputs (results/r9_figures/):
  f1b_umap_joint.csv            N1  per cell: gold-standard X_umap, cell type, class, tissue, ZT, sample
  f1c_marker_expression.csv     N2  scope (all / cochlea / bm) × cell type × candidate marker: mean log1p(CP10k),
                                     fraction of cells with counts > 0, cells; candidates = one canonical marker per
                                     type (literature) + top-3 label markers per type (00_audit/label_markers.csv)
  f1c_dendrogram_merges.csv     N3  average-linkage merges of the 35 cell types on 1 − Pearson r of pseudobulk
                                     log2(CPM + 1) over the 2,000 most variable expressed genes (display ordering only)
  f1c_dendrogram_leaves.csv     N3  leaf order of that tree
  ed1_sample_qc_quantiles.csv   N4  per sample × QC metric: 5 / 25 / 50 / 75 / 95 % quantiles
  ed1_time_intermixing.csv      N5  per tissue × cell type: share of same-type k-nearest neighbours (k = 15, per-tissue
                                     latent from 20_dr) that come from another sample (= another ZT), observed vs
                                     expected under random mixing within the type

Run on the server: python 901_figure_extract.py
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.io import H5AD, LABEL_COLS, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402

MOD = fd.MOD
K_NN = 15
N_HVG = 2000
CANONICAL = {   # one literature marker per gold-standard type (display candidates; specificity is reported, not assumed)
    "Hair Cell (Outer/Inner)": "Ptprq", "Spiral Ganglion Neuron": "Snap25", "Schwann Cell": "Mpz",
    "Oligodendrocyte": "Mag", "SV Marginal Cell": "Kcnq1", "SV Intermediate Cell (Pigmented)": "Dct",
    "SV Intermediate Cell (Functional)": "Kcnj10", "Epithelial Cell (Non-sensory)": "Epcam",
    "Fibrocyte (Spiral Ligament)": "Coch", "Fibrocyte (Cochlear)": "Apod", "Interdental Cell": "Otoa",
    "Osteoprogenitor": "Ptn", "Osteoblast": "Ibsp", "Endothelial Cell": "Cdh5", "HSC / MPP": "Kit",
    "Cycling Cell": "Mki67", "Cycling Cell (Stromal)": "Top2a", "Erythroblast": "Car2", "Pro-B Cell (Cycling)": "Vpreb1a",
    "Pre-B Cell": "Rag1", "B Cell (Transitional)": "Vpreb3", "B Cell (Mature)": "Ms4a1", "B Cell (Naïve)": "Ighd",
    "T Cell (CD3+)": "Cd3e", "NK Cell": "Klrb1c", "Granulocyte-Monocyte Progenitor": "Elane", "Neutrophil (Band)": "Ngp",
    "Neutrophil (Mature)": "Ltf", "Neutrophil (Aged)": "Cxcr2", "Neutrophil (Activated)": "Mmp8",
    "Monocyte (Classical)": "Ly6c2", "Monocyte (Non-classical)": "Nr4a1", "Macrophage": "C1qa", "pDC": "Siglech",
    "Mast Cell": "Cpa3"}
assert set(CANONICAL) == set(fd.TYPE_ORDER)


def prov(df, source, transform):
    return fd.prov(df, source, "", transform)


log(f"reading {H5AD}")
adata = ad.read_h5ad(H5AD)
obs = adata.obs
samples = sample_table(obs).set_index("sample_id")
cell_type = obs["cell_type"].astype(str).to_numpy()
assert set(cell_type) == set(fd.TYPE_ORDER), "label set changed"
tissue = obs["tissue"].astype(str).map(TISSUE_FROM_OBS).to_numpy()
sample = obs["batch_id"].astype(str).to_numpy()
zt = samples.loc[sample, "zt"].to_numpy()
log(f"{adata.n_obs} cells x {adata.n_vars} genes; label hash "
    f"{pd.util.hash_pandas_object(obs[LABEL_COLS].astype(str), index=True).sum()}")
ct_tab = fd.celltypes().set_index("cell_type")

# ---------------------------------------------------------------- N1 joint UMAP (gold standard)
um = np.asarray(adata.obsm["X_umap"], dtype=float)
assert um.shape == (adata.n_obs, 2) and np.isfinite(um).all()
u = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "umap_1": um[:, 0], "umap_2": um[:, 1], "cell_type": cell_type,
                  "abbr": ct_tab.loc[cell_type, "abbr"].to_numpy(), "cell_class": obs["cell_class"].astype(str).to_numpy(),
                  "tissue": tissue, "zt": zt, "sample_id": sample})
write_csv(prov(u, "data/coca_circadian.h5ad (obsm X_umap, obs labels)", "export (no recomputation)"), MOD,
          "f1b_umap_joint.csv", "Fig. 1b / d: gold-standard joint UMAP of all cells with labels, tissue, ZT and sample",
          required=["cell_id", "umap_1", "umap_2", "cell_type", "tissue", "zt", "source_table", "finding_id", "transform"])

# ---------------------------------------------------------------- shared: per-cell normalisation and type indicator
X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
total = np.asarray(X.sum(axis=1)).ravel()
assert (total > 0).all()
types = fd.TYPE_ORDER
t_idx = pd.Categorical(cell_type, categories=types).codes
G = sp.csr_matrix((np.ones(adata.n_obs), (np.arange(adata.n_obs), t_idx)), shape=(adata.n_obs, len(types)))

# ---------------------------------------------------------------- N2 marker expression
lm = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
top3 = (lm.sort_values(["cell_type", "scores"], ascending=[True, False]).groupby("cell_type").head(3))
cands = pd.concat([pd.DataFrame({"gene": list(CANONICAL.values()), "marker_for": list(CANONICAL), "candidate": "canonical"}),
                   pd.DataFrame({"gene": top3["names"], "marker_for": top3["cell_type"], "candidate": "label_top3"})])
genes_all = pd.Index(adata.var_names.astype(str))
missing = sorted(set(cands["gene"]) - set(genes_all))
if missing:
    log(f"candidate markers absent from the matrix (skipped): {missing}")
cands = cands[cands["gene"].isin(genes_all)]
genes = list(dict.fromkeys(cands["gene"]))
Xg = X[:, genes_all.get_indexer(genes)].astype(np.float64)
Xg = sp.diags(1e4 / total) @ Xg
Xg.data = np.log1p(Xg.data)
nz = Xg.copy()
nz.data = np.ones_like(nz.data)
rows = []
for scope, mask in [("all", np.ones(adata.n_obs, bool)), ("cochlea", tissue == "cochlea"), ("bm", tissue == "bm")]:
    Gm = G[mask]
    n = np.asarray(Gm.sum(axis=0)).ravel()
    s = np.asarray((Gm.T @ Xg[mask]).todense())
    f = np.asarray((Gm.T @ nz[mask]).todense())
    with np.errstate(invalid="ignore", divide="ignore"):
        mean, frac = s / n[:, None], f / n[:, None]
    for i, t in enumerate(types):
        for j, g in enumerate(genes):
            rows.append({"scope": scope, "cell_type": t, "gene": g, "n_cells": int(n[i]),
                         "mean_log1p_cp10k": mean[i, j] if n[i] else np.nan, "frac_expressing": frac[i, j] if n[i] else np.nan})
me = pd.DataFrame(rows)
role = {g: "; ".join(f"{c}:{ct_tab.loc[m, 'abbr']}" for c, m in zip(d["candidate"], d["marker_for"]))
        for g, d in cands.groupby("gene")}
me["candidate_for"] = me["gene"].map(role)
write_csv(prov(me, "data/coca_circadian.h5ad (raw counts); 00_audit/label_markers.csv",
               "per-cell log1p(counts / total x 1e4); mean and fraction > 0 per cell type (pooled and per tissue)"), MOD,
          "f1c_marker_expression.csv", "Fig. 1c / ED2: marker expression per cell type (mean log1p CP10k, fraction expressing)",
          required=["scope", "cell_type", "gene", "mean_log1p_cp10k", "frac_expressing", "n_cells",
                    "source_table", "finding_id", "transform"])
log(f"N2: {len(genes)} candidate markers x {len(types)} types x 3 scopes")

# ---------------------------------------------------------------- N3 cell-type dendrogram (display ordering only)
pb = np.asarray((G.T @ X).todense(), dtype=float)                       # types x genes, summed raw counts
cpm = pb / pb.sum(axis=1, keepdims=True) * 1e6
lc = np.log2(cpm + 1)
expressed = lc.mean(axis=0) > 1
var = np.where(expressed, lc.var(axis=0), -np.inf)
hv = np.argsort(var)[::-1][:N_HVG]
r = np.corrcoef(lc[:, hv])
d = 1 - r
np.fill_diagonal(d, 0)
Z = linkage(squareform(d, checks=False), method="average")
coph = float(cophenet(Z, squareform(d, checks=False))[0])
leaves = dendrogram(Z, no_plot=True, labels=types)["ivl"]
ids = list(types) + [f"m{i + 1:02d}" for i in range(len(types) - 1)]
mg = pd.DataFrame({"merge_id": [f"m{i + 1:02d}" for i in range(len(Z))], "left": [ids[int(a)] for a in Z[:, 0]],
                   "right": [ids[int(b)] for b in Z[:, 1]], "height": Z[:, 2], "n_leaves": Z[:, 3].astype(int)})
mg["cophenetic_r"] = coph
mg["n_genes_used"] = len(hv)
write_csv(prov(mg, "data/coca_circadian.h5ad (raw counts)",
               f"pseudobulk sum per type -> log2(CPM+1); top {N_HVG} variable genes with mean > 1; distance 1 - Pearson r; "
               "average linkage"), MOD, "f1c_dendrogram_merges.csv",
          "Fig. 1c: average-linkage merges of the 35 cell types (pseudobulk correlation; display ordering only)",
          required=["merge_id", "left", "right", "height", "source_table", "finding_id", "transform"])
lv = pd.DataFrame({"cell_type": leaves, "leaf_order": range(1, len(leaves) + 1)})
write_csv(prov(lv, "f1c_dendrogram_merges.csv", "scipy dendrogram leaf order"), MOD, "f1c_dendrogram_leaves.csv",
          "Fig. 1c: leaf order of the cell-type dendrogram", required=["cell_type", "leaf_order", "source_table", "finding_id",
                                                                         "transform"])
log(f"N3: tree on {len(hv)} genes; cophenetic r {coph:.3f}")

# ---------------------------------------------------------------- N4 per-sample QC quantiles
qc_cols = [c for c in ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"] if c in obs.columns]
qc_cols += [c for c in obs.columns if "doublet" in c.lower() and pd.api.types.is_numeric_dtype(obs[c])
            and obs[c].nunique() > 2]                         # scores only; 0/1 calls have no useful quantiles
qrows = []
for sid, sub in obs.groupby("batch_id", observed=True):
    for c in qc_cols:
        q = sub[c].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
        qrows.append({"sample_id": sid, "tissue": samples.loc[sid, "tissue"], "zt": samples.loc[sid, "zt"],
                      "platform": samples.loc[sid, "platform"], "metric": c, "n_cells": len(sub),
                      "q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4]})
write_csv(prov(pd.DataFrame(qrows), "data/coca_circadian.h5ad (obs QC metrics)", "per-sample quantiles"), MOD,
          "ed1_sample_qc_quantiles.csv", "ED1: per-sample QC metric quantiles (5/25/50/75/95 %)",
          required=["sample_id", "tissue", "zt", "metric", "q50", "source_table", "finding_id", "transform"])
log(f"N4: QC metrics {qc_cols}")

# ---------------------------------------------------------------- N5 time intermixing within cell types
irows = []
for t_code in fd.TISSUES:
    lat = pd.read_csv(RESULTS / f"20_dr/embedding_{t_code}_latent.csv", index_col="cell_id")
    m = tissue == t_code
    cid = adata.obs_names[m].astype(str)
    assert lat.index.isin(cid).all() and len(lat) == m.sum(), f"{t_code}: latent cells do not match the h5ad"
    lat = lat.loc[cid]
    ct_t, smp_t = cell_type[m], sample[m]
    # kneighbors() without a query returns each point's K_NN neighbours excluding itself
    nbr = NearestNeighbors(n_neighbors=K_NN, n_jobs=-1).fit(lat.to_numpy()).kneighbors(return_distance=False)
    same_type = ct_t[nbr] == ct_t[:, None]
    other_smp = smp_t[nbr] != smp_t[:, None]
    n_same = same_type.sum(axis=1)
    obs_share = np.where(n_same > 0, (same_type & other_smp).sum(axis=1) / np.maximum(n_same, 1), np.nan)
    cnt = pd.crosstab(ct_t, smp_t)
    n_t = cnt.sum(axis=1)
    n_st = cnt.to_numpy()[cnt.index.get_indexer(ct_t), cnt.columns.get_indexer(smp_t)]
    exp_share = 1 - (n_st - 1) / np.maximum(n_t.loc[ct_t].to_numpy() - 1, 1)
    df = pd.DataFrame({"cell_type": ct_t, "obs": obs_share, "exp": exp_share, "n_same": n_same})
    for t, sub in df.groupby("cell_type"):
        ok = sub["n_same"] >= 3
        if ok.sum() < 20:
            continue
        o, e = sub.loc[ok, "obs"].mean(), sub.loc[ok, "exp"].mean()
        irows.append({"tissue": t_code, "cell_type": t, "n_cells": len(sub), "n_cells_scored": int(ok.sum()), "k": K_NN,
                      "mean_same_type_neighbours": sub["n_same"].mean(), "observed_other_sample_share": o,
                      "expected_other_sample_share": e, "intermixing_ratio": o / e if e > 0 else np.nan})
    log(f"N5 {t_code}: {m.sum()} cells, latent {lat.shape[1]} dims")
im = fd.tissue_cols(fd.annotate(pd.DataFrame(irows)))
write_csv(prov(im, "20_dr/embedding_{tissue}_latent.csv; data/coca_circadian.h5ad (labels, samples)",
               f"kNN (k = {K_NN}, Euclidean, per-tissue latent); share of same-type neighbours from another sample; expected "
               "= mean over cells of 1 - (n_sample,type - 1) / (n_type - 1); ratio = observed / expected; types with >= 20 "
               "cells having >= 3 same-type neighbours"), MOD, "ed1_time_intermixing.csv",
          "ED1: within-type intermixing of sampling times (other-sample share of same-type kNN neighbours, observed / expected)",
          required=["tissue", "cell_type", "observed_other_sample_share", "expected_other_sample_share", "intermixing_ratio",
                    "source_table", "finding_id", "transform"])
log("done")
