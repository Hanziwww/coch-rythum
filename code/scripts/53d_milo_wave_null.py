"""Round 9 (D75, O15) — exact time-relabelling null of the Milo production waves (Fig. 5d; M1 / M3 / B02 method lines).

Pseudotime is recomputed exactly as in 630_myeloid_trajectories.py (neutrophil branch: GMP + band / mature / aged /
activated neutrophils, scVI latent, kNN 15, diffusion map, DPT rooted in GMP) and 520_b_identity_states.py (B lineage,
scVI latent, rooted in pro-B); the observed circular-linear correlation of neighbourhood peak ZT with index-cell
pseudotime is asserted to equal the stored statistic (m_milo_wave.csv, b_milo_wave.csv). The statistic is then
recomputed with the neighbourhood peaks refitted under each of the 59 non-identity dihedral relabellings of the six
sampling times (53c_milo_wave_relabel.R); exact p = share of the 60 classes (observed included) with r >= observed. The
'nominal p < 0.1' selection is re-applied within each class.
Outputs (results/53_milo/): milo_wave_relabel_null.csv, milo_wave_null_summary.csv
Run: python scripts/53d_milo_wave_null.py
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, H5AD, RESULTS, TISSUE_FROM_OBS, log, write_csv  # noqa: E402

MOD = "53_milo"
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
ORDER = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)"]
BST = ORDER + ["B Cell (Mature)"]


def dpt(cells, labels, lat, root, term):
    """Diffusion pseudotime as in 630 / 520 (kNN 15 on the latent, 10 diffusion components, root = most extreme cell
    of the root label along the first diffusion component oriented root -> terminal)."""
    T = ad.AnnData(np.zeros((len(cells), 1)), obs=pd.DataFrame({"cell_type": labels}, index=cells))
    T.obsm["X_lat"] = lat
    sc.pp.neighbors(T, use_rep="X_lat", n_neighbors=15, random_state=0)
    sc.tl.diffmap(T, n_comps=10)
    dc1 = T.obsm["X_diffmap"][:, 1]
    ct = np.asarray(labels)
    if dc1[ct == root].mean() > dc1[np.isin(ct, term)].mean():
        dc1 = -dc1
    roots = np.where(ct == root)[0]
    T.uns["iroot"] = int(roots[np.argmin(dc1[roots])])
    sc.tl.dpt(T)
    p_ = T.obs["dpt_pseudotime"].to_numpy(float)
    fin = np.isfinite(p_)
    return pd.Series((p_ - p_[fin].min()) / (p_[fin].max() - p_[fin].min()), index=cells)


def rcl(x, th):
    rc, rs = np.corrcoef(x, np.cos(th))[0, 1], np.corrcoef(x, np.sin(th))[0, 1]
    rcs = np.corrcoef(np.cos(th), np.sin(th))[0, 1]
    return float(np.sqrt(max(0.0, (rc ** 2 + rs ** 2 - 2 * rc * rs * rcs) / (1 - rcs ** 2))))


A = ad.read_h5ad(H5AD, backed="r")
obs = A.obs.copy()
obs["tcode"] = obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
obs["cell_type"] = obs["cell_type"].astype(str)
stored = {"neutrophil": pd.read_csv(RESULTS / "r6_myeloid/m_milo_wave.csv"), "B": pd.read_csv(RESULTS / "r5_bcell/b_milo_wave.csv")}
SEL = {"neutrophil": {"nominal p < 0.1": "nominal p < 0.1", "all": "all neutrophil-lineage neighbourhoods"},
       "B": {"nominal p < 0.1": "nominal p < 0.1", "all": "all B neighbourhoods"}}
rows = []
for t in ["bm", "cochlea"]:
    lat = pd.read_csv(CACHE / f"r2_embedding/latent_{t}_scvi.csv", index_col=0)
    o = obs[obs["tcode"] == t]
    # neutrophil branch exactly as 630: cells of GMP + NEU + MONO in obs order, then the GMP + NEU subset
    om = o[o["cell_type"].isin([GMP] + NEU + ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"])]
    on = om[om["cell_type"].isin([GMP] + NEU)]
    pts = {"neutrophil": dpt(on.index, on["cell_type"].to_numpy(), lat.reindex(on.index).to_numpy(float), GMP, NEU[2:])}
    ob = o[o["cell_type"].isin(BST)]
    pts["B"] = dpt(ob.index, ob["cell_type"].to_numpy(), lat.reindex(ob.index).to_numpy(float), ORDER[0], [ORDER[3]])
    nh = pd.read_csv(RESULTS / f"53_milo/milo_nhoods_{t}_scvi.csv")
    pk = pd.read_csv(RESULTS / f"53_milo/milo_relabel_peaks_{t}_scvi.csv")
    for lin, types in (("neutrophil", [GMP] + NEU), ("B", BST)):
        m = nh[nh["majority_cell_type"].isin(types)][["nhood", "index_cell"]].copy()
        m["pt"] = m["index_cell"].map(pts[lin])
        m = m.dropna(subset=["pt"])
        for k, g in pk.groupby("relabel_class"):
            x = m.merge(g[["nhood", "peak_zt", "p"]], on="nhood")
            for sel, xx in (("nominal p < 0.1", x[x["p"] < 0.1]), ("all", x)):
                if len(xx) < 8:
                    continue
                rows.append({"tissue": t, "lineage": lin, "selection": sel, "relabel_class": int(k), "n_nhoods": len(xx),
                             "circ_linear_r": rcl(xx["pt"].to_numpy(), phase.W * xx["peak_zt"].to_numpy())})
        for sel in ("nominal p < 0.1", "all"):
            obs_r = [r for r in rows if r["tissue"] == t and r["lineage"] == lin and r["selection"] == sel and r["relabel_class"] == 1]
            st = stored[lin]
            st = st[(st["tissue"] == t) & (st["selection"] == SEL[lin][sel])]
            if obs_r and len(st):
                assert abs(obs_r[0]["circ_linear_r"] - float(st["circ_linear_r"].iloc[0])) < 1e-6, (t, lin, sel, obs_r[0], st.to_dict("records"))
                assert obs_r[0]["n_nhoods"] == int(st["n_nhoods"].iloc[0])
    log(f"{t}: waves reproduced and relabelled")
nl = pd.DataFrame(rows)
summ = []
for (t, lin, sel), g in nl.groupby(["tissue", "lineage", "selection"]):
    o_ = g[g["relabel_class"] == 1]
    if not len(o_):
        continue
    r0 = float(o_["circ_linear_r"].iloc[0])
    nn = g[g["relabel_class"] > 1]["circ_linear_r"]
    st = stored[lin]
    st = st[(st["tissue"] == t) & (st["selection"] == SEL[lin][sel])]
    summ.append({"tissue": t, "lineage": lin, "selection": sel, "n_nhoods": int(o_["n_nhoods"].iloc[0]), "observed_r": r0,
                 "null_mean_r": float(nn.mean()), "null_q95_r": float(nn.quantile(0.95)), "n_classes": int(len(g)),
                 "p_exact": float((g["circ_linear_r"] >= r0 - 1e-12).mean()),
                 "stored_p_perm_shuffle_pseudotime": float(st["p_perm"].iloc[0]) if len(st) else np.nan})
write_csv(nl, MOD, "milo_wave_relabel_null.csv", "Milo production waves (scVI neighbourhoods): circular-linear r of "
          "neighbourhood peak ZT with index-cell pseudotime under each of the 60 dihedral relabellings of the sampling times "
          "(class 1 = observed)", required=["tissue", "lineage", "selection", "relabel_class", "circ_linear_r"])
write_csv(pd.DataFrame(summ), MOD, "milo_wave_null_summary.csv", "Milo production waves: observed r vs the exact "
          "time-relabelling null (exact p) next to the original pseudotime-shuffle p", required=["tissue", "lineage",
                                                                                                 "observed_r", "p_exact"])
print(pd.DataFrame(summ).round(4).to_string())
log("done")
