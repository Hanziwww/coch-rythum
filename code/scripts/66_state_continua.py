"""Module 5.2d — receiver-side migratory competence and arrival-state continua (per cell), plus BM-likeness.

Per-cell scores (log1p CP10k; score = mean z of a gene set within tissue, z computed over all cells of the tissue):
  neutrotime   = mature_terminal − immature_granule      (neutrophils/GMP)
  neutro_aging = aging_cxcr4hi_sell_lo set
  cxcr4_cxcr2  = log2((Cxcr4 + 1)/(Cxcr2 + 1)) on CP10k  (retention vs egress; Adrover 2019)
  sell         = log1p CP10k of Sell (CD62L)
  egress_vs_retention = mean z(S1pr1, Klf2) − z(Cd69)   (lymphocytes)
  mono_to_mac  = mean z(H2-Ab1, Adgre1, Mrc1, C1qa, Apoe) − mean z(Ly6c2, Ccr2, Plac8, Chil3)
  b_maturation = mean z(Cr2, Fcer2a, H2-Aa, Cd74, Ms4a1) − mean z(Vpreb1, Vpreb3, Igll1, Dntt, Rag1)
Per sample × cell type: mean and quartiles; cosinor of the per-sample means (60-class permutation p); Wasserstein
distances between consecutive ZTs within a tissue and between tissues at the same ZT (scores standardised per tissue).
BM-likeness (if cache/20_dr/final_joint.h5ad exists): for each cochlear immune cell, the fraction of its 30 nearest
neighbours in the joint embedding that are BM cells, divided by the BM fraction of that cell type (1 = BM-like mixing).

Outputs (results/66_state_continua/): state_scores_by_cell.csv, state_scores_by_sample.csv, state_score_rhythm.csv,
  state_continuum_shifts.csv, bm_likeness_by_cell.csv, bm_likeness_by_sample.csv, bm_likeness_rhythm.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, H5AD, log, sample_table, write_csv  # noqa: E402

MOD = "66_state_continua"
SC = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
NEUT = ["Granulocyte-Monocyte Progenitor", "Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)",
        "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
BLIN = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
LYMPH = ["T Cell (CD3+)", "NK Cell", "B Cell (Naïve)", "B Cell (Mature)", "B Cell (Transitional)", "pDC"]
SCORES = {
    "neutrotime": (NEUT, SC["neutrophil"]["mature_terminal"], SC["neutrophil"]["immature_granule"]),
    "neutro_aging": (NEUT, SC["neutrophil"]["aging_cxcr4hi_sell_lo"], []),
    "egress_vs_retention": (LYMPH, ["S1pr1", "Klf2"], ["Cd69"]),
    "mono_to_mac": (MONO, ["H2-Ab1", "Adgre1", "Mrc1", "C1qa", "Apoe"], ["Ly6c2", "Ccr2", "Plac8", "Chil3"]),
    "b_maturation": (BLIN, ["Cr2", "Fcer2a", "H2-Aa", "Cd74", "Ms4a1"], ["Vpreb1", "Vpreb3", "Igll1", "Dntt", "Rag1"]),
}

a = ad.read_h5ad(H5AD)
st = sample_table(a.obs).set_index("sample_id")
lib = np.asarray(a.X.sum(1)).ravel()
X = (sp.diags(1e4 / np.maximum(lib, 1)).astype(np.float32) @ a.X.tocsr().astype(np.float32)).tocsr()
genes = pd.Index(a.var_names)
obs = a.obs[["batch_id", "cell_type", "tissue"]].copy()
obs["cell_type"] = obs["cell_type"].astype(str)   # avoid categorical groupby over empty combinations
obs["sample_id"] = obs["batch_id"].astype(str)
obs["tissue_code"] = obs["sample_id"].map(st["tissue"])
obs["zt"] = obs["sample_id"].map(st["zt"])
cells = []
for tissue in ["cochlea", "bm"]:
    m = (obs["tissue_code"] == tissue).to_numpy()
    sub = X[m]
    ob = obs[m].copy()
    L = sub.copy()
    L.data = np.log1p(L.data)

    def zmean(gl):
        idx = [genes.get_loc(g) for g in gl if g in genes]
        if not idx:
            return np.zeros(sub.shape[0])
        M = np.asarray(L[:, idx].todense())
        return ((M - M.mean(0)) / (M.std(0) + 1e-6)).mean(1)

    for name, (types, up, down) in SCORES.items():
        sel = ob["cell_type"].isin(types).to_numpy()
        v = zmean(up) - (zmean(down) if down else 0)
        ob[name] = np.where(sel, v, np.nan)
    def gene(g):
        return np.asarray(sub[:, genes.get_loc(g)].todense()).ravel() if g in genes else np.zeros(sub.shape[0])
    sel = ob["cell_type"].isin(NEUT).to_numpy()
    ob["cxcr4_cxcr2"] = np.where(sel, np.log2((gene("Cxcr4") + 1) / (gene("Cxcr2") + 1)), np.nan)
    ob["sell"] = np.where(ob["cell_type"].isin(NEUT + LYMPH).to_numpy(), np.log1p(gene("Sell")), np.nan)
    cells.append(ob)
cells = pd.concat(cells)
score_cols = list(SCORES) + ["cxcr4_cxcr2", "sell"]
out_cells = cells[["tissue_code", "sample_id", "zt", "cell_type"] + score_cols].copy()
out_cells.index.name = "cell_id"
out_cells = out_cells.dropna(how="all", subset=score_cols)
write_csv(out_cells.reset_index(), MOD, "state_scores_by_cell.csv", "Per-cell migratory/arrival-state scores",
          required=["cell_id", "tissue_code", "sample_id", "zt", "cell_type"])

long = out_cells.reset_index().melt(id_vars=["cell_id", "tissue_code", "sample_id", "zt", "cell_type"],
                                    value_vars=score_cols, var_name="score", value_name="value").dropna()
agg = (long.groupby(["tissue_code", "cell_type", "score", "sample_id", "zt"])["value"]
       .agg(n="size", mean="mean", q25=lambda x: x.quantile(0.25), median="median", q75=lambda x: x.quantile(0.75))
       .reset_index())
write_csv(agg, MOD, "state_scores_by_sample.csv", "Per-sample summaries of state scores per cell type",
          required=["tissue_code", "cell_type", "score", "sample_id", "zt", "mean"])
rh = []
for (t, c, s), g in agg.groupby(["tissue_code", "cell_type", "score"]):
    g = g.sort_values("zt")
    if len(g) < 6 or g["n"].min() < 10:
        continue
    y = g["mean"].to_numpy()[:, None]
    f = cosinor.fit(y, g["zt"].to_numpy())
    _, _, pp, _ = cosinor.permutation_test(y, g["zt"].to_numpy())
    rh.append({"tissue": t, "cell_type": c, "score": s, "amplitude": f["amp"][0], "peak_zt": f["peak"][0],
               "p": f["p"][0], "p_perm": pp[0]})
write_csv(pd.DataFrame(rh), MOD, "state_score_rhythm.csv", "24-h rhythm of per-sample mean state scores",
          required=["tissue", "cell_type", "score", "peak_zt", "p", "p_perm"])

shifts = []
for (c, s), g in long.groupby(["cell_type", "score"]):
    for t, gt in g.groupby("tissue_code"):
        zts = sorted(gt["zt"].unique())
        for i, z in enumerate(zts):
            z2 = zts[(i + 1) % len(zts)]
            v1, v2 = gt[gt["zt"] == z]["value"], gt[gt["zt"] == z2]["value"]
            if len(v1) >= 20 and len(v2) >= 20:
                shifts.append({"comparison": "consecutive_zt", "tissue": t, "cell_type": c, "score": s,
                               "zt_a": z, "zt_b": z2, "wasserstein": wasserstein_distance(v1, v2),
                               "mean_shift": v2.mean() - v1.mean()})
    for z, gz in g.groupby("zt"):
        v1, v2 = gz[gz["tissue_code"] == "cochlea"]["value"], gz[gz["tissue_code"] == "bm"]["value"]
        if len(v1) >= 20 and len(v2) >= 20:
            shifts.append({"comparison": "cochlea_vs_bm", "tissue": "both", "cell_type": c, "score": s,
                           "zt_a": z, "zt_b": z, "wasserstein": wasserstein_distance(v1, v2),
                           "mean_shift": v1.mean() - v2.mean()})
write_csv(pd.DataFrame(shifts), MOD, "state_continuum_shifts.csv",
          "Wasserstein distances of state-score distributions between consecutive ZTs and between tissues",
          required=["comparison", "cell_type", "score", "wasserstein"])

fj = CACHE / "20_dr" / "final_joint.h5ad"
if fj.exists():
    j = ad.read_h5ad(fj)
    conn = j.obsp["distances"].tocsr()
    is_bm = (j.obs["tissue_code"] == "bm").to_numpy()
    nn_bm = np.asarray([is_bm[conn.indices[conn.indptr[i]:conn.indptr[i + 1]]].mean() for i in range(j.n_obs)])
    df = j.obs[["tissue_code", "sample_id", "zt", "cell_type"]].astype(
        {"tissue_code": str, "sample_id": str, "cell_type": str}).copy()
    df["zt"] = pd.to_numeric(df["zt"])
    df["frac_bm_neighbours"] = nn_bm
    bm_frac = df.groupby("cell_type")["tissue_code"].apply(lambda s: (s == "bm").mean())
    df["bm_likeness"] = df["frac_bm_neighbours"] / df["cell_type"].map(bm_frac).astype(float).clip(lower=1e-3)
    co = df[(df["tissue_code"] == "cochlea") & df["cell_type"].isin(NEUT + MONO + BLIN + LYMPH + ["Mast Cell"])]
    co.index.name = "cell_id"
    write_csv(co.reset_index(), MOD, "bm_likeness_by_cell.csv",
              "Cochlear immune cells: fraction of BM cells among 30 joint-embedding neighbours (normalised)",
              required=["cell_id", "cell_type", "bm_likeness"])
    bs = co.groupby(["cell_type", "sample_id", "zt"])["bm_likeness"].agg(["size", "mean", "median"]).reset_index()
    write_csv(bs, MOD, "bm_likeness_by_sample.csv", "Per-sample BM-likeness of cochlear immune cells",
              required=["cell_type", "sample_id", "zt", "mean"])
    rr = []
    for c, g in bs.groupby("cell_type"):
        g = g.sort_values("zt")
        if len(g) < 6 or g["size"].min() < 10:
            continue
        Y = g[["mean"]].to_numpy()
        f = cosinor.fit(Y, g["zt"].to_numpy())
        _, _, pp, _ = cosinor.permutation_test(Y, g["zt"].to_numpy())
        rr.append({"cell_type": c, "min_cells": int(g["size"].min()), "mesor": f["mesor"][0], "amplitude": f["amp"][0],
                   "peak_zt": f["peak"][0], "p": f["p"][0], "p_perm": pp[0]})
    write_csv(pd.DataFrame(rr), MOD, "bm_likeness_rhythm.csv",
              "24-h rhythm of the per-sample mean BM-likeness of cochlear immune cells (cosinor, 60-class permutation)",
              required=["cell_type", "peak_zt", "p", "p_perm"])
else:
    log("final_joint.h5ad not found — BM-likeness skipped (run after 23_dr_finalize.py)")
log("done")
