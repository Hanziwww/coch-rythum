"""Round 5, WP-B2 — B-cell identity continuum and state alternation (H2 method 2, H3), BM and cochlea (equal weight).

Cells: the five gold-standard B labels (labels unchanged; everything here is a new per-cell column).
(1) Trajectory (D30): kNN (15) on the unsupervised scVI latent (cache/r2_embedding/latent_<tissue>_scvi.csv, the
    C__scvi_none_nl30 family) → diffusion map → DPT rooted at the pro-B cell with the lowest DC1 after orienting DC1
    from pro-B to naïve. Sensitivities: selected scANVI latent (20_dr) and a B-only PCA (2,000 HVGs, 30 PCs).
    Check: Spearman of pseudotime with stage order (pro 0 … naïve 3, mature 4).
    Per sample: pseudotime quantiles of the lineage and the median pseudotime within each stage (position inside a
    label = continuous identity), cosinor with the exact 60-class permutation p and LOTO.
(2) Milo wave (H2, method 2): scVI Milo neighbourhoods (53, *_scvi) with a B-stage majority; index-cell pseudotime vs
    neighbourhood peak ZT (nominal p < 0.1): circular–linear correlation (Mardia) with a pseudotime-permutation p; mean
    peak per pseudotime quintile.
(3) State axes (H3; D31 external sets; IEG and Cd83 excluded):
      immgen_FrF_vs_FrE   ImmGen recirculating-mature (FrF) up-set minus immature (FrE) down-set (560; microarray first)
      mhcII_GO            MSigDB GO:BP antigen processing and presentation of exogenous peptide via MHC class II
      curated_recirc_vs_immature  recirculation / LN homing (Ccr7, Sell, S1pr1, Cr2, Fcer2a, Ighd, Cxcr5) minus BM-resident
                          immature (Cd93, Cd24a, Vpreb3, Rag1, Rag2, Dntt, Igll1, Myb)
    Pseudobulk (primary for H3): per primary cell type of the tissue, log2 CPM, gene z across the 6 samples, axis =
    mean z(up) − mean z(down); within-sample centring = minus the per-sample median over the tissue's primary types
    (as 43); depth-matched rerun (binomial thinning of each type's 6 libraries to the smallest, as 41 DEPTH_MATCH).
    H3 primary = immgen_FrF_vs_FrE in the pooled naïve + mature B pseudobulk, centred: exact permutation p and peak
    inside ZT12–24. Cell level (agreeing method): scanpy score_genes(up) − score_genes(down) per cell; per sample ×
    stage median; all cells and high-confidence cells (00_audit margin > 0).
(4) Identity drift (agreeing method for H3): within-lineage stage classifier (logistic regression on 50 PCs of B-lineage
    HVGs, 5-fold stratified CV; gold labels as targets, never changed); per cell the out-of-fold posterior of the next
    stage (naïve → mature; mature → naïve); per sample × stage mean, cosinor + permutation.
(5) Immunoglobulin transcripts (exploratory): log2((Ighd+1)/(Ighm+1)), log2((Igkc+1)/(Σ Iglc+1)), plasma-cell score
    (Jchain, Xbp1, Prdm1, Sdc1, Mzb1) per cell; per sample × stage mean, cosinor + permutation.
Every rhythm row carries r with the tissue-wide sample pattern (210 PC1) and with the per-sample median UMIs.
Outputs (results/r5_bcell/): b_pseudotime_checks.csv, b_pseudotime_by_sample.csv, b_pseudotime_rhythm.csv, b_milo_wave.csv,
b_state_axes_by_sample.csv, b_state_axes_rhythm.csv, b_identity_drift_rhythm.csv, b_ig_rhythm.csv, b_primary_h3.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import CACHE, EXTERNAL, H5AD, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402

MOD = "r5_bcell"
ORDER = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)"]
BST = ORDER + ["B Cell (Mature)"]
RANK = {s: i for i, s in enumerate(BST)}
CLASSES = cosinor.dihedral_classes(6)
rng = np.random.default_rng(20)
IEG = {"Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36", "Dusp1", "Klf2", "Klf4",
       "Klf6", "Btg2", "Ppp1r15a", "Gadd45b", "Socs3", "Cebpd", "Cd83", "Cd69", "Nr4a2", "Nr4a3"}
CUR_UP = ["Ccr7", "Sell", "S1pr1", "Cr2", "Fcer2a", "Ighd", "Cxcr5"]
CUR_DN = ["Cd93", "Cd24a", "Vpreb3", "Rag1", "Rag2", "Dntt", "Igll1", "Myb"]


def rhythm(y, zt):
    f = cosinor.fit(np.asarray(y, float)[:, None], zt)
    obs = float(f["F"][0])
    null = [float(cosinor.fit(np.asarray(y, float)[:, None], zt[list(p)])["F"][0]) for p in CLASSES[1:]]
    return {"amplitude": float(f["amp"][0]), "peak_zt": float(f["peak"][0]), "peak_clock": float(phase.zt_to_clock(f["peak"][0])),
            "p": float(f["p"][0]), "p_perm": (1 + sum(x >= obs for x in null)) / len(CLASSES),
            "loto_phase_sd_h": float(cosinor.loto(np.asarray(y, float)[:, None], zt)["loto_phase_sd_h"][0])}


def gmt(path, name):
    for line in open(path):
        x = line.rstrip("\n").split("\t")
        if x[0] == name:
            return x[2:]
    return []


sig = pd.read_csv(RESULTS / "r5_bcell/immgen_b_signatures.csv")
ref = "ImmGen microarray (GSE15907)" if (sig["reference"] == "ImmGen microarray (GSE15907)").any() else sig["reference"].iloc[0]
ax = sig[(sig["reference"] == ref) & (sig["signature"] == "axis:FrF_vs_FrE")]
AXES = {"immgen_FrF_vs_FrE": (ax.loc[ax["direction"].str.startswith("up"), "gene"].tolist(), ax.loc[ax["direction"].str.startswith("down"), "gene"].tolist()),
        "mhcII_GO": ([g for g in gmt(EXTERNAL / "resources/msigdb_m5_gobp.gmt",
                                     "GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_EXOGENOUS_PEPTIDE_ANTIGEN_VIA_MHC_CLASS_II") if g not in IEG], []),
        "curated_recirc_vs_immature": (CUR_UP, CUR_DN)}
log("axes: " + "; ".join(f"{k} up {len(u)} / down {len(d)}" for k, (u, d) in AXES.items()))

A = ad.read_h5ad(H5AD)
A.obs["tissue_code"] = A.obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
st_all = sample_table(A.obs)
qc = pd.read_csv(RESULTS / "00_audit/label_support_qc.csv", usecols=["cell_id", "margin"]).set_index("cell_id")["margin"]
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
pb_design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")

checks, pt_samp, pt_rh, wave, axs_s, axs_r, drift, igr, prim = [], [], [], [], [], [], [], [], []
for t in ["bm", "cochlea"]:
    st = st_all[st_all["tissue"] == t].sort_values("zt")
    zt = st["zt"].to_numpy(float)
    samples = st["sample_id"].tolist()
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    B = A[(A.obs["tissue_code"] == t) & A.obs["cell_type"].astype(str).isin(BST)].copy()
    B.obs["stage"] = B.obs["cell_type"].astype(str)
    B.obs["sample_id"] = B.obs["batch_id"].astype(str)
    B.layers["counts"] = B.X.copy()
    sc.pp.normalize_total(B, target_sum=1e4)
    sc.pp.log1p(B)
    umi = pd.Series(np.asarray(B.layers["counts"].sum(1)).ravel(), index=B.obs_names)
    depth = umi.groupby(B.obs["sample_id"]).median().reindex(samples).to_numpy(float)
    log(f"{t}: {B.n_obs} B-lineage cells")

    def add_ctx(r, y):
        r["r_with_sample_pattern_pc1"] = float(np.corrcoef(y, pc1)[0, 1])
        r["r_with_median_umis"] = float(np.corrcoef(y, depth)[0, 1])
        return r

    # ---------------- (1) trajectories
    lat = {"scVI (unsupervised; primary)": pd.read_csv(CACHE / f"r2_embedding/latent_{t}_scvi.csv", index_col=0),
           "scANVI (selected; sensitivity)": pd.read_csv(RESULTS / f"20_dr/embedding_{t}_latent.csv", index_col=0)}
    Bp = B.copy()
    sc.pp.highly_variable_genes(Bp, n_top_genes=2000, flavor="seurat_v3", layer="counts")
    lat["B-only PCA (sensitivity)"] = pd.DataFrame(PCA(30, random_state=0).fit_transform(Bp[:, Bp.var["highly_variable"]].X.toarray()),
                                                   index=B.obs_names)
    pts = {}
    for name, L in lat.items():
        X = L.reindex(B.obs_names).to_numpy(float)
        T = ad.AnnData(np.zeros((B.n_obs, 1)), obs=B.obs[["stage"]].copy())
        T.obsm["X_lat"] = X
        sc.pp.neighbors(T, use_rep="X_lat", n_neighbors=15, random_state=0)
        sc.tl.diffmap(T, n_comps=10)
        dc1 = T.obsm["X_diffmap"][:, 1]
        if dc1[T.obs["stage"] == ORDER[0]].mean() > dc1[T.obs["stage"] == ORDER[3]].mean():
            dc1 = -dc1
        pro = np.where(T.obs["stage"] == ORDER[0])[0]
        T.uns["iroot"] = int(pro[np.argmin(dc1[pro])])
        sc.tl.dpt(T)
        p_ = T.obs["dpt_pseudotime"].to_numpy(float)
        p_ = (p_ - np.nanmin(p_[np.isfinite(p_)])) / (np.nanmax(p_[np.isfinite(p_)]) - np.nanmin(p_[np.isfinite(p_)]))
        pts[name] = pd.Series(p_, index=B.obs_names)
        rho = stats.spearmanr(B.obs["stage"].map(RANK), p_, nan_policy="omit")[0]
        rho4 = stats.spearmanr(B.obs.loc[B.obs["stage"].isin(ORDER), "stage"].map(RANK), pd.Series(p_, index=B.obs_names)[B.obs["stage"].isin(ORDER)],
                               nan_policy="omit")[0]
        checks.append({"tissue": t, "embedding": name, "n_cells": B.n_obs, "spearman_stage_order_all5": rho, "spearman_stage_order_pro_to_naive": rho4,
                       "median_pseudotime_by_stage": "; ".join(f"{s.split(' (')[0]} {np.nanmedian(p_[B.obs['stage'] == s]):.2f}" for s in BST)})
    names = list(pts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            checks.append({"tissue": t, "embedding": f"{names[i]} vs {names[j]}", "n_cells": B.n_obs,
                           "spearman_stage_order_all5": stats.spearmanr(pts[names[i]], pts[names[j]], nan_policy="omit")[0]})
    for name, p_ in pts.items():
        df = B.obs[["sample_id", "stage"]].assign(pt=p_.to_numpy())
        rows = {}
        for s in samples:
            d = df[df["sample_id"] == s]
            rec = {"tissue": t, "embedding": name, "sample_id": s, "zt": float(st.set_index("sample_id").loc[s, "zt"])}
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                rec[f"lineage_q{int(q * 100)}"] = float(d["pt"].quantile(q))
            for stg in BST:
                rec[f"median_within:{stg}"] = float(d.loc[d["stage"] == stg, "pt"].median())
            rows[s] = rec
            pt_samp.append(rec)
        tab = pd.DataFrame([rows[s] for s in samples])
        for col in [c for c in tab.columns if c.startswith(("lineage_q", "median_within"))]:
            y = tab[col].to_numpy(float)
            if np.isnan(y).any():
                continue
            pt_rh.append(add_ctx({"tissue": t, "embedding": name, "feature": col, **rhythm(y, zt)}, y))
    # ---------------- (2) Milo wave on scVI neighbourhoods
    nh = pd.read_csv(RESULTS / f"53_milo/milo_nhoods_{t}_scvi.csv")
    da = pd.read_csv(RESULTS / f"53_milo/milo_da_results_{t}_scvi.csv")
    m = nh.merge(da[["nhood", "peak_zt", "p", "spatial_fdr"]], on="nhood")
    m = m[m["majority_cell_type"].isin(BST)].copy()
    m["pt"] = m["index_cell"].map(pts["scVI (unsupervised; primary)"])
    for sel_name, mm in [("nominal p < 0.1", m[m["p"] < 0.1]), ("all B neighbourhoods", m)]:
        mm = mm.dropna(subset=["pt"])
        if len(mm) < 8:
            continue
        x, th = mm["pt"].to_numpy(), phase.W * mm["peak_zt"].to_numpy()

        def rcl(x_, th_):
            rc, rs = np.corrcoef(x_, np.cos(th_))[0, 1], np.corrcoef(x_, np.sin(th_))[0, 1]
            rcs = np.corrcoef(np.cos(th_), np.sin(th_))[0, 1]
            return np.sqrt(max(0.0, (rc ** 2 + rs ** 2 - 2 * rc * rs * rcs) / (1 - rcs ** 2)))

        obs = rcl(x, th)
        null = np.array([rcl(rng.permutation(x), th) for _ in range(5000)])
        q = pd.qcut(mm["pt"], 5, labels=False, duplicates="drop")
        wave.append({"tissue": t, "selection": sel_name, "n_nhoods": len(mm), "circ_linear_r": obs, "p_perm": float((1 + (null >= obs).sum()) / 5001),
                     "peak_by_pseudotime_quintile": "; ".join(f"Q{int(k) + 1} ZT{phase.circ_mean(g['peak_zt'].to_numpy())[0]:.1f} (n {len(g)})"
                                                              for k, g in mm.groupby(q))})
    # ---------------- (3) state axes — pseudobulk (all primary types, for centring) and cell level
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    d0 = pb_design[(pb_design["tissue"] == t) & (pb_design["level"] == "cell_type") & (pb_design["status"] == "primary")]
    types = sorted(d0["group"].unique())
    groups = {ty: [ty] for ty in types}
    groups["B naïve + mature (pooled)"] = ["B Cell (Naïve)", "B Cell (Mature)"]

    def axis_scores(counts_by_group, up, dn):
        out = {}
        for gname, M in counts_by_group.items():
            L = np.log2(M / M.sum(0) * 1e6 + 1)
            L = L[(M / M.sum(0) * 1e6).mean(1) >= 1]
            Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
            u = Z.reindex([g for g in up if g in Z.index]).mean(0)
            v = Z.reindex([g for g in dn if g in Z.index]).mean(0) if dn else 0
            out[gname] = (u - v).reindex(samples).to_numpy(float)
        return out

    raw = {g: pb[[f"{x}|{s}" for s in samples for x in members]].T.groupby([s for s in samples for _ in members]).sum().T.reindex(columns=samples)
           if len(members) > 1 else pb[[f"{members[0]}|{s}" for s in samples]].set_axis(samples, axis=1)
           for g, members in groups.items()}
    thin = {}
    for g, M in raw.items():
        lib = M.sum(0)
        k = lib.min()
        thin[g] = M.apply(lambda col: pd.Series(rng.binomial(col.astype(int).to_numpy(), k / col.sum()), index=col.index))
    for axname, (up, dn) in AXES.items():
        for version, cnts in [("primary", raw), ("depth_matched", thin)]:
            sc_ = axis_scores(cnts, up, dn)
            med = np.nanmedian(np.vstack([sc_[ty] for ty in types]), axis=0)
            for g in [x for x in BST + ["B naïve + mature (pooled)"] if x in sc_]:
                for kind, y in [("uncentred", sc_[g]), ("centred", sc_[g] - med)]:
                    if np.isnan(y).any():
                        continue
                    for s_, v_ in zip(samples, y):
                        axs_s.append({"tissue": t, "axis": axname, "level": "pseudobulk", "version": version, "kind": kind, "group": g,
                                      "sample_id": s_, "zt": float(st.set_index("sample_id").loc[s_, "zt"]), "score": float(v_)})
                    axs_r.append(add_ctx({"tissue": t, "axis": axname, "level": "pseudobulk", "version": version, "kind": kind, "group": g,
                                          "n_up": len(up), "n_down": len(dn), **rhythm(y, zt)}, y))
        # cell level
        upg, dng = [g for g in up if g in B.var_names], [g for g in dn if g in B.var_names]
        sc.tl.score_genes(B, upg, score_name="_up", random_state=0)
        if dng:
            sc.tl.score_genes(B, dng, score_name="_dn", random_state=0)
            cell = B.obs["_up"] - B.obs["_dn"]
        else:
            cell = B.obs["_up"]
        for subset, mask in [("all cells", np.ones(B.n_obs, bool)), ("high-confidence cells", (qc.reindex(B.obs_names) > 0).to_numpy())]:
            df = B.obs.loc[mask, ["sample_id", "stage"]].assign(score=cell[mask].to_numpy())
            for g_name, gstages in [(s_, [s_]) for s_ in BST] + [("B naïve + mature (pooled)", ["B Cell (Naïve)", "B Cell (Mature)"])]:
                y = df[df["stage"].isin(gstages)].groupby("sample_id")["score"].median().reindex(samples).to_numpy(float)
                if np.isnan(y).any():
                    continue
                axs_r.append(add_ctx({"tissue": t, "axis": axname, "level": f"cell ({subset})", "version": "primary", "kind": "median per sample",
                                      "group": g_name, "n_up": len(upg), "n_down": len(dng), **rhythm(y, zt)}, y))
    # ---------------- (4) identity drift classifier
    hv = Bp.var["highly_variable"].to_numpy()
    Xp = PCA(50, random_state=0).fit_transform(B[:, hv].X.toarray())
    yl = B.obs["stage"].map(RANK).to_numpy()
    post = np.zeros((B.n_obs, len(BST)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xp, yl):
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(Xp[tr], yl[tr])
        post[te] = clf.predict_proba(Xp[te])
    nxt = {0: 1, 1: 2, 2: 3, 3: 4, 4: 3}
    nx = post[np.arange(B.n_obs), [nxt[k] for k in yl]]
    df = B.obs[["sample_id", "stage"]].assign(next_post=nx, own_post=post[np.arange(B.n_obs), yl])
    for stg in BST:
        y = df[df["stage"] == stg].groupby("sample_id")["next_post"].mean().reindex(samples).to_numpy(float)
        own = df[df["stage"] == stg]["own_post"].mean()
        if np.isnan(y).any():
            continue
        drift.append(add_ctx({"tissue": t, "stage": stg, "next_stage": BST[nxt[RANK[stg]]], "mean_own_posterior": float(own),
                              "mean_next_posterior": float(np.mean(y)), **rhythm(y, zt)}, y))
    # ---------------- (5) immunoglobulin transcripts
    def gx(gl):
        g = [x for x in gl if x in B.var_names]
        return np.asarray(B[:, g].layers["counts"].sum(1)).ravel() / umi.to_numpy() * 1e4 if g else np.zeros(B.n_obs)

    ig = pd.DataFrame({"ighd_ighm": np.log2((gx(["Ighd"]) + 1) / (gx(["Ighm"]) + 1)),
                       "kappa_lambda": np.log2((gx(["Igkc"]) + 1) / (gx(["Iglc1", "Iglc2", "Iglc3"]) + 1))}, index=B.obs_names)
    sc.tl.score_genes(B, [g for g in ["Jchain", "Xbp1", "Prdm1", "Sdc1", "Mzb1"] if g in B.var_names], score_name="plasma", random_state=0)
    ig["plasma_score"] = B.obs["plasma"]
    ig = ig.join(B.obs[["sample_id", "stage"]])
    for feat in ["ighd_ighm", "kappa_lambda", "plasma_score"]:
        for stg in BST:
            y = ig[ig["stage"] == stg].groupby("sample_id")[feat].mean().reindex(samples).to_numpy(float)
            if np.isnan(y).any():
                continue
            igr.append(add_ctx({"tissue": t, "feature": feat, "stage": stg, "mean": float(np.mean(y)), **rhythm(y, zt)}, y))
    # ---------------- H3 primary
    r_ = [r for r in axs_r if r["tissue"] == t and r["axis"] == "immgen_FrF_vs_FrE" and r["level"] == "pseudobulk" and r["version"] == "primary"
          and r["kind"] == "centred" and r["group"] == "B naïve + mature (pooled)"][0]
    prim.append({"hypothesis": "H3 state alternation", "tissue": t,
                 "primary_statistic": "exact permutation p of the centred ImmGen FrF-vs-FrE axis in the pooled naïve + mature B pseudobulk",
                 "primary_p": r_["p_perm"], "peak_zt": r_["peak_zt"], "peak_clock": r_["peak_clock"], "pre_specified_window": "ZT12–24",
                 "in_window": bool(12 <= r_["peak_zt"] < 24),
                 "detail": f"amplitude {r_['amplitude']:.2f}; LOTO SD {r_['loto_phase_sd_h']:.2f} h; r with sample pattern {r_['r_with_sample_pattern_pc1']:+.2f}"})
    log(f"{t}: H3 p_perm {r_['p_perm']:.3f} peak ZT{r_['peak_zt']:.1f}")

write_csv(pd.DataFrame(checks), MOD, "b_pseudotime_checks.csv", "B-lineage pseudotime: stage-order Spearman per embedding and agreement between embeddings",
          required=["tissue", "embedding", "spearman_stage_order_all5"])
write_csv(pd.DataFrame(pt_samp), MOD, "b_pseudotime_by_sample.csv", "B-lineage pseudotime quantiles and within-stage medians per sample (ZT)",
          required=["tissue", "embedding", "sample_id", "zt"])
write_csv(pd.DataFrame(pt_rh), MOD, "b_pseudotime_rhythm.csv", "Rhythms of pseudotime quantiles / within-stage positions (cosinor, exact permutation, LOTO)",
          required=["tissue", "embedding", "feature", "peak_zt", "peak_clock", "p_perm"])
write_csv(pd.DataFrame(wave), MOD, "b_milo_wave.csv", "Milo (scVI) B neighbourhoods: peak ZT vs pseudotime (circular-linear r, permutation p; quintile peaks)",
          required=["tissue", "selection", "n_nhoods", "circ_linear_r", "p_perm"])
write_csv(pd.DataFrame(axs_s), MOD, "b_state_axes_by_sample.csv", "B state-axis scores per sample (pseudobulk; uncentred / centred; primary / depth-matched)",
          required=["tissue", "axis", "group", "sample_id", "zt", "score"])
write_csv(pd.DataFrame(axs_r), MOD, "b_state_axes_rhythm.csv",
          "B state-axis rhythms (external sets: ImmGen FrF-vs-FrE, GO MHC-II, curated recirculation vs immature) at pseudobulk and cell level",
          required=["tissue", "axis", "level", "group", "peak_zt", "peak_clock", "p_perm"])
write_csv(pd.DataFrame(drift), MOD, "b_identity_drift_rhythm.csv", "Identity drift: next-stage posterior of a within-lineage classifier per stage, rhythm",
          required=["tissue", "stage", "next_stage", "peak_zt", "p_perm"])
write_csv(pd.DataFrame(igr), MOD, "b_ig_rhythm.csv", "Immunoglobulin transcript ratios (IgD/IgM, κ/λ) and plasma score per stage, rhythm",
          required=["tissue", "feature", "stage", "peak_zt", "p_perm"])
write_csv(pd.DataFrame(prim), MOD, "b_primary_h3.csv", "Primary hypothesis H3 (state alternation): primary statistic per tissue",
          required=["hypothesis", "tissue", "primary_p", "in_window"])
ar = pd.DataFrame(axs_r)
log("\n" + ar[(ar["group"].isin(["B naïve + mature (pooled)", "B Cell (Naïve)", "B Cell (Mature)"])) & (ar["version"] == "primary")]
    [["tissue", "axis", "level", "kind", "group", "peak_zt", "p", "p_perm", "r_with_sample_pattern_pc1"]].round(3).to_string())
log("\n" + pd.DataFrame(checks).round(3).to_string())
log("\n" + pd.DataFrame(wave).round(3).to_string())
