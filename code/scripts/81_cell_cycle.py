"""Module 6.3 — rhythmic proliferation / haematopoiesis.

Cell-cycle scoring per cell (scanpy score_genes_cell_cycle with mouse orthologues of the Tirosh et al. S and G2/M
lists, computed within tissue on log1p CP10k), phase assignment (G1 / S / G2M). Per sample × cell type: fraction in
S, in G2M and cycling (S + G2M), mean scores. Rhythm of the logit cycling fraction (cosinor, 60-class permutation,
LOTO) per tissue × cell type with >= 20 cells in every sample. Clock–cycle coupling: correlation across the six ZTs
between the cycling fraction and pseudobulk Wee1 / Nr1d1 / Ccnb1 log2CPM of the same cell type.

Outputs (results/81_cell_cycle/): cell_cycle_by_sample.csv, cell_cycle_rhythm.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, H5AD, RESULTS, log, sample_table, write_csv  # noqa: E402

MOD = "81_cell_cycle"
S_GENES = ["Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1", "Ung", "Gins2", "Mcm6", "Cdca7", "Dtl", "Prim1",
           "Uhrf1", "Mlf1ip", "Cenpu", "Hells", "Rfc2", "Rpa2", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2",
           "Ubr7", "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc6", "Exo1", "Tipin", "Dscc1", "Blm",
           "Casp8ap2", "Usp1", "Clspn", "Pola1", "Chaf1b", "Brip1", "E2f8"]
G2M_GENES = ["Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80", "Cks2", "Nuf2", "Cks1b", "Mki67",
             "Tmpo", "Cenpf", "Tacc3", "Fam64a", "Pimreg", "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11",
             "Anp32e", "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Hn1", "Jpt1", "Cdc20", "Ttk", "Cdc25c", "Kif2c",
             "Rangap1", "Ncapd2", "Dlgap5", "Cdca2", "Cdca8", "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln",
             "Lbr", "Ckap5", "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa"]

a = ad.read_h5ad(H5AD)
st = sample_table(a.obs).set_index("sample_id")
lib = np.asarray(a.X.sum(1)).ravel()
X = (sp.diags(1e4 / np.maximum(lib, 1)).astype(np.float32) @ a.X.tocsr().astype(np.float32)).tocsr()
X.data = np.log1p(X.data)
res = []
for tissue, obs_val in [("cochlea", "cochlea"), ("bm", "bone marrow")]:
    m = (a.obs["tissue"] == obs_val).to_numpy()
    b = ad.AnnData(X[m], obs=a.obs.loc[m, ["batch_id", "cell_type"]].copy(), var=pd.DataFrame(index=a.var_names))
    sc.tl.score_genes_cell_cycle(b, s_genes=[g for g in S_GENES if g in b.var_names],
                                 g2m_genes=[g for g in G2M_GENES if g in b.var_names])
    b.obs["tissue"] = tissue
    res.append(b.obs[["tissue", "batch_id", "cell_type", "S_score", "G2M_score", "phase"]])
cc = pd.concat(res)
cc["sample_id"] = cc["batch_id"].astype(str)
cc["cell_type"] = cc["cell_type"].astype(str)   # avoid categorical groupby over empty combinations
cc["zt"] = cc["sample_id"].map(st["zt"])
agg = cc.groupby(["tissue", "cell_type", "sample_id", "zt"]).agg(
    n=("phase", "size"), frac_S=("phase", lambda x: (x == "S").mean()), frac_G2M=("phase", lambda x: (x == "G2M").mean()),
    mean_S_score=("S_score", "mean"), mean_G2M_score=("G2M_score", "mean")).reset_index()
agg["frac_cycling"] = agg["frac_S"] + agg["frac_G2M"]
write_csv(agg, MOD, "cell_cycle_by_sample.csv", "Cell-cycle phase fractions and scores per sample × cell type",
          required=["tissue", "cell_type", "sample_id", "zt", "n", "frac_cycling"])

rows = []
for (t, c), g in agg.groupby(["tissue", "cell_type"]):
    g = g.sort_values("zt")
    if len(g) < 6 or g["n"].min() < 20:
        continue
    k = g["frac_cycling"] * g["n"]
    y = np.log((k + 0.5) / (g["n"] - k + 0.5)).to_numpy()[:, None]
    zt = g["zt"].to_numpy()
    f = cosinor.fit(y, zt)
    _, _, pp, _ = cosinor.permutation_test(y, zt)
    lo = cosinor.loto(y, zt)
    row = {"tissue": t, "cell_type": c, "mean_frac_cycling": g["frac_cycling"].mean(), "amp_logit": f["amp"][0],
           "peak_zt": f["peak"][0], "p": f["p"][0], "p_perm": pp[0], "loto_phase_sd_h": lo["loto_phase_sd_h"][0]}
    try:
        pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet",
                             columns=[f"{c}|{s}" for s in g["sample_id"]])
        lc = np.log2(pb / pb.sum(0) * 1e6 + 1)
        for gene in ["Wee1", "Nr1d1", "Ccnb1", "Mki67"]:
            if gene in lc.index:
                row[f"corr_frac_cycling_vs_{gene}"] = float(np.corrcoef(g["frac_cycling"], lc.loc[gene].to_numpy())[0, 1])
    except Exception:
        pass
    rows.append(row)
write_csv(pd.DataFrame(rows), MOD, "cell_cycle_rhythm.csv",
          "24-h rhythm of the cycling fraction per tissue × cell type and clock–cycle correlations",
          required=["tissue", "cell_type", "mean_frac_cycling", "peak_zt", "p", "p_perm"])
log("done")
