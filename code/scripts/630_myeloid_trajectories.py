"""Round 6, WP-M3 — myeloid trajectories and fates (M3 primary; M5 agreeing), BM and cochlea paired (D42).

Cells: GMP, the four neutrophil states, classical / non-classical monocytes, macrophages (labels unchanged).
Latent: unsupervised scVI (cache/r2_embedding/latent_<tissue>_scvi.csv; C__scvi_none_nl30), kNN 15, diffusion map.
(1) Neutrophil branch: DPT rooted at the GMP with the lowest DC1 after orienting DC1 from GMP to aged / activated
    neutrophils. Check: Spearman with GMP 0, band 1, mature 2, aged / activated 3.
    M3 primary: per-sample median branch pseudotime of GMP + band + mature cells (granulopoietic part), cosinor with the
    exact 60-class permutation p (and LOTO); the whole distribution by the OT rhythm test (lib/disttest).
(2) Monocyte branch: DPT rooted at GMP over GMP + classical + non-classical monocytes + macrophages; per-sample median
    pseudotime of monocytes / macrophages (M5 agreeing).
(3) CellRank (PseudotimeKernel on the joint myeloid DPT, soft threshold; GPCCA estimator with terminal states set from
    labels: Neutrophil = aged + activated, Macrophage, Non-classical monocyte): fate probabilities per cell;
    per sample the median neutrophil fate of GMPs (M3 agreeing) and the median macrophage fate of classical monocytes
    (M5 agreeing), with permutation p.
(4) Palantir (sensitivity; subsample ≤ 8,000 cells per tissue): pseudotime correlation with DPT and per-sample
    neutrophil branch probability of GMPs.
(5) Milo wave: scVI Milo neighbourhoods (53) with a neutrophil-lineage majority — circular-linear correlation of the
    neighbourhood peak ZT with index-cell pseudotime (permutation), quintile peaks.
Packages cellrank / palantir are loaded from envs/traj_pkgs (isolated; scipy pinned there).
Outputs (results/r6_myeloid/): m_trajectory_checks.csv, m_trajectory_by_sample.csv, m_trajectory_rhythm.csv,
m_milo_wave.csv, m_primary_m3.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, "envs/traj_pkgs")
import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(1, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.abundance import loto_sd, ols, perm_p  # noqa: E402
from lib.disttest import ot_rhythm_test  # noqa: E402
from lib.io import CACHE, H5AD, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402

MOD = "r6_myeloid"
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
ORD = {GMP: 0, NEU[0]: 1, NEU[1]: 2, NEU[2]: 3, NEU[3]: 3}
rng = np.random.default_rng(63)
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")


def dpt(o_df, root_label, term_labels, lat):
    """o_df: obs rows (cells) of the branch; lat: their latent coordinates (same order)."""
    T = ad.AnnData(np.zeros((len(o_df), 1)), obs=o_df[["cell_type", "sample_id"]].copy())
    T.obsm["X_lat"] = lat
    sc.pp.neighbors(T, use_rep="X_lat", n_neighbors=15, random_state=0)
    sc.tl.diffmap(T, n_comps=10)
    dc1 = T.obsm["X_diffmap"][:, 1]
    ct = T.obs["cell_type"].astype(str).to_numpy()
    if dc1[ct == root_label].mean() > dc1[np.isin(ct, term_labels)].mean():
        dc1 = -dc1
    roots = np.where(ct == root_label)[0]
    T.uns["iroot"] = int(roots[np.argmin(dc1[roots])])
    sc.tl.dpt(T)
    p_ = T.obs["dpt_pseudotime"].to_numpy(float)
    fin = np.isfinite(p_)
    p_ = (p_ - p_[fin].min()) / (p_[fin].max() - p_[fin].min())
    T.obs["pt"] = p_
    return T


def rhythm_row(tissue, feature, y, zt, pc1, values=None):
    F, p, pk, amp = ols(y, zt)
    r = {"tissue": tissue, "feature": feature, "peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p,
         "p_perm": perm_p(y, zt), "loto_phase_sd_h": loto_sd(y, zt), "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1])}
    if values is not None:
        r.update(ot_rhythm_test(values))
    return r


A = ad.read_h5ad(H5AD, backed="r")
obs = A.obs.copy()
obs["tcode"] = obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
obs["sample_id"] = obs["batch_id"].astype(str)
st_all = sample_table(A.obs)
checks, bys, rh, wave, prim = [], [], [], [], []
import cellrank as cr  # noqa: E402
import palantir  # noqa: E402

for t in ["bm", "cochlea"]:
    st = st_all[st_all["tissue"] == t].sort_values("zt")
    samples, zt = st["sample_id"].tolist(), st["zt"].to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    lat_all = pd.read_csv(CACHE / f"r2_embedding/latent_{t}_scvi.csv", index_col=0)
    o = obs[(obs["tcode"] == t) & obs["cell_type"].astype(str).isin([GMP] + NEU + MONO)].copy()
    o["cell_type"] = o["cell_type"].astype(str)
    lat = lat_all.reindex(o.index).to_numpy(float)
    # ---------------- (1) neutrophil branch
    on = o[o["cell_type"].isin([GMP] + NEU)]
    Tn = dpt(on, GMP, NEU[2:], lat_all.reindex(on.index).to_numpy(float))
    rho = stats.spearmanr(Tn.obs["cell_type"].map(ORD), Tn.obs["pt"], nan_policy="omit")[0]
    checks.append({"tissue": t, "branch": "neutrophil (GMP → aged / activated)", "n_cells": Tn.n_obs, "spearman_with_expected_order": rho,
                   "median_pseudotime_by_type": "; ".join(f"{c} {Tn.obs.loc[Tn.obs['cell_type'] == c, 'pt'].median():.2f}" for c in [GMP] + NEU)})
    gran = Tn.obs[Tn.obs["cell_type"].isin([GMP] + NEU[:2])]
    vals = [gran.loc[gran["sample_id"] == s, "pt"].to_numpy() for s in samples]
    y = np.array([np.nanmedian(v) for v in vals])
    r = rhythm_row(t, "neutrophil-branch pseudotime, GMP + band + mature (median) [M3 primary]", y, zt, pc1, vals)
    rh.append(r)
    prim.append({"hypothesis": "M3 granulopoiesis timing", "tissue": t, "primary_statistic": "exact permutation p of the per-sample median "
                 "neutrophil-branch pseudotime (scVI DPT) of GMP + band + mature cells", "primary_p": r["p_perm"], "peak_zt": r["peak_zt"],
                 "peak_clock": r["peak_clock"], "detail": f"OT ratio {r['ot_ratio']:.2f} (p {r['ot_p_perm']:.3f}); LOTO SD {r['loto_phase_sd_h']:.2f} h"})
    for c in [GMP] + NEU:
        vals_c = [Tn.obs.loc[(Tn.obs["sample_id"] == s) & (Tn.obs["cell_type"] == c), "pt"].to_numpy() for s in samples]
        if min(len(v) for v in vals_c) < 10:
            continue
        rh.append(rhythm_row(t, f"neutrophil-branch pseudotime within {c} (median)", np.array([np.nanmedian(v) for v in vals_c]), zt, pc1, vals_c))
    for s, v in zip(samples, vals):
        bys.append({"tissue": t, "sample_id": s, "zt": float(st.set_index("sample_id").loc[s, "zt"]), "feature": "neutrophil-branch pseudotime GMP+band+mature",
                    "median": float(np.nanmedian(v)), "q25": float(np.nanquantile(v, 0.25)), "q75": float(np.nanquantile(v, 0.75)), "n_cells": len(v)})
    # ---------------- (2) monocyte branch
    om = o[o["cell_type"].isin([GMP] + MONO)]
    Tm = dpt(om, GMP, ["Macrophage"], lat_all.reindex(om.index).to_numpy(float))
    checks.append({"tissue": t, "branch": "monocyte (GMP → macrophage)", "n_cells": Tm.n_obs,
                   "spearman_with_expected_order": stats.spearmanr(Tm.obs["cell_type"].map({GMP: 0, MONO[0]: 1, MONO[1]: 2, MONO[2]: 2}), Tm.obs["pt"], nan_policy="omit")[0],
                   "median_pseudotime_by_type": "; ".join(f"{c} {Tm.obs.loc[Tm.obs['cell_type'] == c, 'pt'].median():.2f}" for c in [GMP] + MONO)})
    mm = Tm.obs[Tm.obs["cell_type"].isin(MONO)]
    vals_m = [mm.loc[mm["sample_id"] == s, "pt"].to_numpy() for s in samples]
    rh.append(rhythm_row(t, "monocyte-branch pseudotime, monocytes + macrophages (median) [M5 agreeing]",
                         np.array([np.nanmedian(v) for v in vals_m]), zt, pc1, vals_m))
    # ---------------- (3) CellRank on the joint myeloid DPT
    Tj = dpt(o, GMP, NEU[2:] + ["Macrophage"], lat)
    Tj.obs["cell_type"] = Tj.obs["cell_type"].astype("category")
    pk_ = cr.kernels.PseudotimeKernel(Tj, time_key="pt").compute_transition_matrix(threshold_scheme="soft", show_progress_bar=False)
    g = cr.estimators.GPCCA(pk_)
    ct = Tj.obs["cell_type"].astype(str)
    term = {"Neutrophil": list(Tj.obs_names[ct.isin(NEU[2:])]), "Macrophage": list(Tj.obs_names[ct == "Macrophage"]),
            "NonClassicalMono": list(Tj.obs_names[ct == "Monocyte (Non-classical)"])}
    g.set_terminal_states(term)
    g.compute_fate_probabilities(show_progress_bar=False)
    fp = pd.DataFrame(np.asarray(g.fate_probabilities), index=Tj.obs_names, columns=list(g.fate_probabilities.names))
    for cell, fate, lab in [(GMP, "Neutrophil", "CellRank neutrophil fate of GMPs (median) [M3 agreeing]"),
                            ("Monocyte (Classical)", "Macrophage", "CellRank macrophage fate of classical monocytes (median) [M5 agreeing]")]:
        sel = Tj.obs_names[ct == cell]
        vals_f = [fp.loc[sel[Tj.obs.loc[sel, "sample_id"] == s], fate].to_numpy() for s in samples]
        if min(len(v) for v in vals_f) >= 10:
            rh.append(rhythm_row(t, lab, np.array([np.nanmedian(v) for v in vals_f]), zt, pc1, vals_f))
    checks.append({"tissue": t, "branch": "CellRank fates (mean by type)", "n_cells": Tj.n_obs,
                   "median_pseudotime_by_type": "; ".join(f"{c}: " + ", ".join(f"{f} {fp.loc[Tj.obs_names[ct == c], f].mean():.2f}" for f in fp.columns)
                                                          for c in [GMP, MONO[0]])})
    # ---------------- (4) Palantir sensitivity (subsample)
    idx = rng.choice(o.index.to_numpy(), size=min(8000, len(o)), replace=False)
    sub = o.loc[idx]
    P = ad.AnnData(np.zeros((len(sub), 1)), obs=sub[["cell_type", "sample_id"]].copy())
    P.obsm["X_lat"] = lat_all.reindex(sub.index).to_numpy(float)
    dm = palantir.utils.run_diffusion_maps(pd.DataFrame(P.obsm["X_lat"], index=P.obs_names), n_components=10)
    ms_ = palantir.utils.determine_multiscale_space(dm)
    ctp = P.obs["cell_type"].astype(str)
    early = Tj.obs.loc[Tj.obs_names.isin(P.obs_names) & (ct == GMP).to_numpy()].sort_values("pt").index[0]
    terms = pd.Series({Tj.obs.loc[Tj.obs_names.isin(P.obs_names) & ct.isin(NEU[2:]).to_numpy()].sort_values("pt").index[-1]: "Neutrophil",
                       Tj.obs.loc[Tj.obs_names.isin(P.obs_names) & (ct == "Macrophage").to_numpy()].sort_values("pt").index[-1]: "Macrophage"})
    pr = palantir.core.run_palantir(ms_, early, terminal_states=terms, num_waypoints=500, n_jobs=1, use_early_cell_as_start=True)
    rho_p = stats.spearmanr(pr.pseudotime.reindex(P.obs_names), Tj.obs.loc[P.obs_names, "pt"], nan_policy="omit")[0]
    bp = pr.branch_probs.rename(columns=lambda c_: terms.get(c_, c_)).reindex(P.obs_names)   # columns may be terminal cell ids
    gm = P.obs_names[ctp == GMP]
    vals_p = [bp.loc[gm[P.obs.loc[gm, "sample_id"] == s], "Neutrophil"].to_numpy() for s in samples]
    checks.append({"tissue": t, "branch": "Palantir (sensitivity, subsample)", "n_cells": len(sub), "spearman_with_expected_order": rho_p,
                   "median_pseudotime_by_type": "Spearman with joint DPT in the column to the left"})
    if min(len(v) for v in vals_p) >= 5:
        rh.append(rhythm_row(t, "Palantir neutrophil branch probability of GMPs (median; sensitivity)", np.array([np.nanmedian(v) for v in vals_p]), zt, pc1))
    # ---------------- (5) Milo wave (scVI neighbourhoods; neutrophil lineage)
    nh = pd.read_csv(RESULTS / f"53_milo/milo_nhoods_{t}_scvi.csv")
    da = pd.read_csv(RESULTS / f"53_milo/milo_da_results_{t}_scvi.csv")
    m = nh.merge(da[["nhood", "peak_zt", "p"]], on="nhood")
    m = m[m["majority_cell_type"].isin([GMP] + NEU)].copy()
    m["pt"] = m["index_cell"].map(Tn.obs["pt"])
    for sel_name, x in [("nominal p < 0.1", m[m["p"] < 0.1]), ("all neutrophil-lineage neighbourhoods", m)]:
        x = x.dropna(subset=["pt"])
        if len(x) < 8:
            continue
        th = phase.W * x["peak_zt"].to_numpy()

        def rcl(v, th_):
            rc, rs = np.corrcoef(v, np.cos(th_))[0, 1], np.corrcoef(v, np.sin(th_))[0, 1]
            rcs = np.corrcoef(np.cos(th_), np.sin(th_))[0, 1]
            return np.sqrt(max(0.0, (rc ** 2 + rs ** 2 - 2 * rc * rs * rcs) / (1 - rcs ** 2)))

        obs_r = rcl(x["pt"].to_numpy(), th)
        null = np.array([rcl(rng.permutation(x["pt"].to_numpy()), th) for _ in range(5000)])
        q = pd.qcut(x["pt"], 5, labels=False, duplicates="drop")
        wave.append({"tissue": t, "selection": sel_name, "n_nhoods": len(x), "circ_linear_r": obs_r, "p_perm": float((1 + (null >= obs_r).sum()) / 5001),
                     "peak_by_pseudotime_quintile": "; ".join(f"Q{int(k) + 1} ZT{phase.circ_mean(gg['peak_zt'].to_numpy())[0]:.1f} (n {len(gg)})"
                                                              for k, gg in x.groupby(q))})
    log(f"{t}: trajectories done; neutrophil-branch Spearman {rho:.2f}; M3 p {prim[-1]['primary_p']:.3f}")

write_csv(pd.DataFrame(checks), MOD, "m_trajectory_checks.csv", "Myeloid trajectories: stage-order checks (DPT), CellRank fate means, Palantir agreement",
          required=["tissue", "branch", "n_cells"])
write_csv(pd.DataFrame(bys), MOD, "m_trajectory_by_sample.csv", "Neutrophil-branch pseudotime per sample (median, quartiles)",
          required=["tissue", "sample_id", "zt", "feature", "median"])
write_csv(pd.DataFrame(rh), MOD, "m_trajectory_rhythm.csv", "Rhythms of trajectory positions and fate probabilities (cosinor, exact permutation, LOTO, OT distribution test)",
          required=["tissue", "feature", "peak_zt", "peak_clock", "p_perm"])
write_csv(pd.DataFrame(wave), MOD, "m_milo_wave.csv", "Milo (scVI) neutrophil-lineage neighbourhoods: peak ZT vs pseudotime", required=["tissue", "selection", "circ_linear_r", "p_perm"])
write_csv(pd.DataFrame(prim), MOD, "m_primary_m3.csv", "Primary hypothesis M3 (granulopoiesis timing): primary statistic per tissue", required=["hypothesis", "tissue", "primary_p"])
log("\n" + pd.DataFrame(checks).round(3).to_string())
log("\n" + pd.DataFrame(rh).round(3).to_string())
log("\n" + pd.DataFrame(wave).round(3).to_string())
