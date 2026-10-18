"""Round 2, WP3 — external reference for temporal-bone marrow: Kolabas et al. 2023 (GSE192616; mouse scRNA-seq of
skull, vertebra, pelvis, femur, humerus, scapula marrow, meninges and brain; naive / sham / MCAO).

Label transfer (our gold standard → their cells): shared genes, log-normalised (1e4), our BM highly variable genes
(seurat flavour, 3000), PCA (50) fitted on our temporal-bone marrow and applied to their cells, distance-weighted kNN
(k = 15) on the PCs; prediction confidence = weighted vote share. Their naive and sham marrow samples only.
(a) Cross-annotation: their level-2 labels × our transferred labels (external check of our annotation).
(b) Composition per bone in our label space (confident calls, vote >= 0.6) vs the mean composition of our six
    temporal-bone marrow samples: Spearman correlation and Aitchison distance (CLR, pseudo-count 0.5 %).
(c) Neutrophil maturation per bone: shares of band / mature / aged / activated within neutrophils vs ours.
Outputs (results/r2_external/): skull_marrow_label_transfer.csv, skull_marrow_composition.csv,
skull_marrow_similarity.csv, skull_marrow_neutrophil_states.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import EXTERNAL, H5AD, log, write_csv  # noqa: E402

MOD = "r2_external"
a = ad.read_h5ad(H5AD)
ours = a[(a.obs["tissue"] == "bone marrow").to_numpy()].copy()
del a
th = ad.read_h5ad(EXTERNAL / "geo/GSE192616/GSE192616_sc_adata_oct22m.h5ad")
th = th[th.obs["condition"].isin(["Naive", "Sham"]).to_numpy() & ~th.obs["region"].isin(["Brain", "Meninges"]).to_numpy()].copy()
th.X = th.layers["raw_counts"]
genes = ours.var_names.intersection(th.var_names)
ours, th = ours[:, genes].copy(), th[:, genes].copy()
for x in (ours, th):
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
sc.pp.highly_variable_genes(ours, n_top_genes=3000, flavor="seurat")
hv = ours.var_names[ours.var["highly_variable"]]
Xo = ours[:, hv].X.toarray(); Xt = th[:, hv].X.toarray()
mu = Xo.mean(0)
pca = PCA(n_components=50, random_state=0).fit(Xo - mu)
Zo, Zt = pca.transform(Xo - mu), pca.transform(Xt - mu)
knn = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(Zo, ours.obs["cell_type"].astype(str).to_numpy())
proba = knn.predict_proba(Zt)
th.obs["pred"] = knn.classes_[proba.argmax(1)]
th.obs["vote"] = proba.max(1)
log(f"transferred labels to {th.n_obs} cells; median vote {np.median(th.obs['vote']):.2f}")

lt = pd.crosstab(th.obs["level2"].astype(str), th.obs["pred"]).stack().rename("n_cells").reset_index()
lt = lt[lt["n_cells"] > 0]
lt["frac_of_their_label"] = lt["n_cells"] / lt.groupby("level2")["n_cells"].transform("sum")
write_csv(lt.rename(columns={"level2": "their_level2", "pred": "our_label"}), MOD, "skull_marrow_label_transfer.csv",
          "Kolabas 2023 level-2 labels × our transferred gold-standard labels", required=["their_level2", "our_label", "n_cells"])

conf = th.obs[th.obs["vote"] >= 0.6]
comp = conf.groupby(["region", "pred"]).size().rename("n").reset_index()
comp["frac"] = comp["n"] / comp.groupby("region")["n"].transform("sum")
ob = ours.obs.groupby(["batch_id", "cell_type"], observed=True).size().rename("n").reset_index()
ob["frac"] = ob["n"] / ob.groupby("batch_id")["n"].transform("sum")
ours_mean = ob.groupby("cell_type")["frac"].mean()
comp_w = comp.pivot(index="pred", columns="region", values="frac").fillna(0)
labels = sorted(set(comp_w.index) | set(ours_mean.index))
comp_w = comp_w.reindex(labels).fillna(0)
comp_w["temporal_bone_marrow_this_atlas"] = ours_mean.reindex(labels).fillna(0)
write_csv(comp_w.reset_index().rename(columns={"index": "cell_type", "pred": "cell_type"}), MOD, "skull_marrow_composition.csv",
          "Composition (our label space) of each bone's marrow (Kolabas 2023, naive+sham) and of our temporal-bone marrow",
          required=["cell_type"])
sim = []
ref = comp_w["temporal_bone_marrow_this_atlas"].to_numpy() + 0.005
clr_ref = np.log(ref) - np.log(ref).mean()
for reg in [c for c in comp_w.columns if c != "temporal_bone_marrow_this_atlas"]:
    v = comp_w[reg].to_numpy() + 0.005
    clr = np.log(v) - np.log(v).mean()
    sim.append({"region": reg, "n_cells_confident": int((conf["region"] == reg).sum()),
                "spearman_vs_temporal_bone": float(stats.spearmanr(v, ref).statistic),
                "aitchison_distance": float(np.linalg.norm(clr - clr_ref))})
write_csv(pd.DataFrame(sim).sort_values("aitchison_distance"), MOD, "skull_marrow_similarity.csv",
          "Similarity of each bone's marrow composition to our temporal-bone marrow (our label space)",
          required=["region", "spearman_vs_temporal_bone", "aitchison_distance"])
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
ns = comp_w.loc[comp_w.index.intersection(NEU)]
ns = ns / ns.sum(0)
write_csv(ns.reset_index().rename(columns={"index": "neutrophil_state", "pred": "neutrophil_state"}), MOD,
          "skull_marrow_neutrophil_states.csv", "Neutrophil maturation-state shares per bone vs our temporal-bone marrow",
          required=["neutrophil_state"])
log("done")
