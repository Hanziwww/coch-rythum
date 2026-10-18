"""Round 3, WP3 — deconvolution of the public cochlear GeoMx data (GSE312646; spiral-ganglion-neuron and supporting-cell
AOIs, noise vs control) with our cochlear atlas.

Cell-profile matrix: per cochlear cell type, mean counts per cell over our six samples (pseudobulk sums / cells),
scaled to 1e4 per cell, restricted to informative genes (top 40 specific markers per type from
00_audit/label_markers.csv, log2FC >= 1.5, adj. p < 1e-10, >= 30 % in-type) measured in the GeoMx panel.
Per AOI: non-negative least squares of the AOI expression on the profiles with rows weighted by 1/sqrt(gene mean)
(variance stabilisation of count-like data; SpatialDecon-style), giving abundance scores β_k ≥ 0 and proportions
β_k / Σβ; fit quality (R² on the weighted scale). Stability: 200 gene-bootstrap refits → 90 % intervals of proportions.
Comparisons per cell type: region (neuron vs supporting AOIs, controls) and noise effect within each region (Welch t,
BH); positive controls: Spiral Ganglion Neuron / Schwann profiles higher in neuron AOIs, hair-cell / epithelial in
supporting-cell AOIs.
Outputs (results/r3_spatial/): geomx_deconvolution_aoi.csv, geomx_deconvolution_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r3_spatial"
rng = np.random.default_rng(320)
gm = pd.read_csv(EXTERNAL / "geo/GSE312646/GSE312646_GeoMX_table.csv.gz", header=0, low_memory=False)
gm = gm.loc[:, ~gm.columns.str.startswith("Unnamed")]
gm = gm.iloc[4:].dropna(subset=["TargetName"]).drop_duplicates("TargetName").set_index("TargetName")
gm = gm.apply(pd.to_numeric, errors="coerce").dropna()
aoi = pd.DataFrame({"aoi": gm.columns})
aoi["region"] = np.where(aoi["aoi"].str.startswith("Neu"), "spiral_ganglion_neurons", "supporting_cells")
aoi["noise"] = np.where(aoi["aoi"].str.contains("_Nos_"), "noise", "control")

design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
d0 = design[(design["tissue"] == "cochlea") & (design["level"] == "cell_type")]
pb = pd.read_parquet(CACHE / "30_pseudobulk/cochlea__cell_type.parquet")
prof = {}
for ct, g in d0.groupby("group"):
    cols = [f"{ct}|{s}" for s in g["sample_id"] if f"{ct}|{s}" in pb.columns]
    if not cols:
        continue
    tot = pb[cols].sum(1)
    ncell = g["n_cells"].sum()
    if ncell < 50:
        continue
    per_cell = tot / ncell
    prof[ct] = per_cell / per_cell.sum() * 1e4
P = pd.DataFrame(prof)
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 1.5) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.3)]
genes = sorted(set(mk.sort_values("scores", ascending=False).groupby("cell_type").head(40)["names"]) & set(gm.index) & set(P.index))
P = P.loc[genes]
Y = gm.loc[genes]
wt = 1 / np.sqrt(Y.mean(1).clip(lower=1e-3)).to_numpy()
log(f"{len(genes)} informative genes, {P.shape[1]} cell profiles, {Y.shape[1]} AOIs")


def fit(yv, Pm, w):
    b, _ = nnls(Pm * w[:, None], yv * w)
    pred = Pm @ b
    ss_res = (((yv - pred) * w) ** 2).sum()
    ss_tot = (((yv - yv.mean()) * w) ** 2).sum()
    return b, 1 - ss_res / ss_tot


rows = []
Pm = P.to_numpy()
for j, a_ in enumerate(Y.columns):
    yv = Y[a_].to_numpy()
    b, r2 = fit(yv, Pm, wt)
    boots = []
    for _ in range(200):
        i = rng.integers(0, len(genes), len(genes))
        bb, _ = fit(yv[i], Pm[i], wt[i])
        boots.append(bb / max(bb.sum(), 1e-12))
    boots = np.array(boots)
    for k, ct in enumerate(P.columns):
        rows.append({"aoi": a_, "region": aoi.loc[j, "region"], "noise": aoi.loc[j, "noise"], "cell_type": ct,
                     "beta": float(b[k]), "proportion": float(b[k] / max(b.sum(), 1e-12)),
                     "prop_q05": float(np.quantile(boots[:, k], 0.05)), "prop_q95": float(np.quantile(boots[:, k], 0.95)),
                     "fit_r2_weighted": float(r2)})
dv = pd.DataFrame(rows)
cls = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
dv["cell_class"] = dv["cell_type"].map(cls)
write_csv(dv, MOD, "geomx_deconvolution_aoi.csv", "GeoMx AOI deconvolution with our cochlear cell profiles (weighted NNLS; "
          "proportions with gene-bootstrap 90 % intervals)", required=["aoi", "region", "noise", "cell_type", "proportion"])
summ = []
for ct, g in dv.groupby("cell_type"):
    c_neu = g[(g["region"] == "spiral_ganglion_neurons") & (g["noise"] == "control")]["proportion"]
    c_sup = g[(g["region"] == "supporting_cells") & (g["noise"] == "control")]["proportion"]
    r = {"cell_type": ct, "cell_class": cls.get(ct, ""), "mean_prop_neuron_control": c_neu.mean(),
         "mean_prop_supporting_control": c_sup.mean(), "p_region": stats.ttest_ind(c_neu, c_sup, equal_var=False).pvalue}
    for reg in ["spiral_ganglion_neurons", "supporting_cells"]:
        a1 = g[(g["region"] == reg) & (g["noise"] == "noise")]["proportion"]
        a0 = g[(g["region"] == reg) & (g["noise"] == "control")]["proportion"]
        r[f"noise_diff_{reg}"] = a1.mean() - a0.mean()
        r[f"p_noise_{reg}"] = stats.ttest_ind(a1, a0, equal_var=False).pvalue if a1.std() > 0 or a0.std() > 0 else np.nan
    summ.append(r)
sm = pd.DataFrame(summ)
for c in [c for c in sm.columns if c.startswith("p_")]:
    m = sm[c].notna()
    sm[c.replace("p_", "q_")] = np.nan
    sm.loc[m, c.replace("p_", "q_")] = multipletests(sm.loc[m, c], method="fdr_bh")[1]
sm["median_fit_r2"] = float(dv.drop_duplicates("aoi")["fit_r2_weighted"].median())
write_csv(sm.sort_values("mean_prop_neuron_control", ascending=False), MOD, "geomx_deconvolution_summary.csv",
          "GeoMx deconvolution: cell-type proportions by region (controls) and noise effects within region",
          required=["cell_type", "mean_prop_neuron_control", "mean_prop_supporting_control"])
log("\n" + sm.round(3).head(20).to_string())
