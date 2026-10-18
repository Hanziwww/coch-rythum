"""Round 6, WP-M4 (part 2) — myeloid functional readiness (M4), myeloid clocks (M7), regulators (exploratory).

(1) M4 primary (pre-specified global test): curated myeloid_function programmes (scripts/resources/curated_sets.yaml:
    primary / secondary / tertiary granules, NETs, NOX2 ROS, chemotaxis receptors, IL-1 / NLRP3, phagocytosis-lysosome,
    type-I IFN) plus Hallmark glycolysis and OXPHOS, scored in the pooled neutrophil + monocyte pseudobulk of each sample
    (mean gene z, log2 CPM, >= 3 expressed members). Statistic = number of programmes with exact permutation p <= 0.05;
    null = the same count under the 59 non-identity relabelings applied jointly to all programmes → exact p.
    Versions: uncentred (primary), within-sample centred against the tissue's primary types (as 43), depth-matched
    (binomial thinning). Per-state programme rhythms (each neutrophil label, classical / non-classical monocytes,
    macrophages) and the Rayleigh clustering of programme peaks across states (agreeing evidence).
(2) M7 primary: cell-level clock projection (332 method) of pooled myeloid cells (GMP, neutrophils, monocytes,
    macrophages; cells with >= 3 clock UMIs): offset consistency over the 6 samples (resultant) with the exact 60-class
    permutation p; offset vs pooled structural cells; power (clock UMIs per cell, usable share). Also neutrophils and
    monocytes / macrophages separately. Pseudobulk clock offsets per myeloid unit with gene-bootstrap CIs and the paired
    difference vs the tissue pool (530 method).
(3) Regulators (exploratory): CollecTRI activity rhythms (41 uncentred, 43 centred) for myeloid regulators and TF
    activity – mRNA lags.
Outputs (results/r6_myeloid/): m_function_rhythm.csv, m_function_global.csv, m_clock_cell_level.csv, m_clock_offsets.csv,
m_tf_activity.csv, m_primary_function_clock.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.abundance import CLASSES, loto_sd, ols, perm_p  # noqa: E402
from lib.io import CACHE, EXTERNAL, H5AD, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402

MOD = "r6_myeloid"
rng = np.random.default_rng(64)
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
CAN = phase.CANONICAL_CLOCK
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
hm = {}
for line in open(EXTERNAL / "resources/msigdb_mh_hallmark.gmt"):
    x = line.rstrip("\n").split("\t")
    hm[x[0]] = x[2:]
PROG = {**{f"myeloid:{k}": v for k, v in SETS["myeloid_function"].items()},
        "hallmark:GLYCOLYSIS": hm["HALLMARK_GLYCOLYSIS"], "hallmark:OXIDATIVE_PHOSPHORYLATION": hm["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]}
TFS = ["Cebpa", "Cebpb", "Cebpe", "Spi1", "Gfi1", "Irf8", "Klf4", "Nr4a1", "Maf", "Mafb", "Nfil3", "Hif1a", "Stat3", "Irf5", "Runx1", "Egr2"]
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")

fr, gl, prim, ck = [], [], [], []
for t in ["bm", "cochlea"]:
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    s0 = d0.drop_duplicates("sample_id").sort_values("zt")
    samples, zt = s0["sample_id"].tolist(), s0["zt"].to_numpy(float)
    prim_types = sorted(d0.loc[d0["status"] == "primary", "group"].unique())
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)

    def mat(types):
        ok = [ty for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples)]
        return sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in ok)

    units = {ty: mat([ty]) for ty in prim_types}
    units["neutrophils + monocytes (pooled)"] = mat(NEU + MONO[:2])
    for st_ in NEU + MONO:
        if st_ not in units:
            units[st_] = mat([st_])

    def thin(M):
        k = M.sum(0).min()
        return M.apply(lambda c: pd.Series(rng.binomial(c.astype(int).to_numpy(), k / c.sum()), index=c.index))

    def scores(U):
        out = {}
        for u, M in U.items():
            cpm = M / M.sum(0) * 1e6
            L = np.log2(cpm + 1)[cpm.mean(1) >= 1]
            Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
            for gs, genes in PROG.items():
                g = [x for x in genes if x in Z.index]
                if len(g) >= 3:
                    out[(u, gs)] = Z.loc[g].mean(0).to_numpy(float)
        return out

    for version, U in [("primary", units), ("depth_matched", {u: thin(M) for u, M in units.items()})]:
        S = scores(U)
        for kind in ["uncentred", "centred"]:
            Y = {}
            for gs in PROG:
                vals = [S[(u, gs)] for u in prim_types if (u, gs) in S]
                med = np.nanmedian(np.vstack(vals), axis=0) if vals else None
                for u in ["neutrophils + monocytes (pooled)"] + NEU + MONO:
                    if (u, gs) not in S:
                        continue
                    y = S[(u, gs)] - (med if kind == "centred" and med is not None else 0)
                    if np.isnan(y).any():
                        continue
                    Y[(u, gs)] = y
                    F, p, pk, amp = ols(y, zt)
                    fr.append({"tissue": t, "version": version, "kind": kind, "unit": u, "programme": gs, "peak_zt": pk,
                               "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p, "p_perm": perm_p(y, zt),
                               "loto_phase_sd_h": loto_sd(y, zt), "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1])})
            # global test on the pooled unit
            pooled = np.vstack([Y[k] for k in Y if k[0] == "neutrophils + monocytes (pooled)"])
            names = [k[1] for k in Y if k[0] == "neutrophils + monocytes (pooled)"]
            Fs = np.vstack([cosinor.fit(pooled.T, zt[list(p_)])["F"] for p_ in CLASSES])
            obs_p = (1 + (Fs[1:] >= Fs[0][None, :]).sum(0)) / len(CLASSES)
            n_obs = int((obs_p <= 0.05).sum())
            null = [int(((1 + (np.delete(Fs, k + 1, 0) >= Fs[k + 1][None, :]).sum(0)) / len(CLASSES) <= 0.05).sum()) for k in range(len(CLASSES) - 1)]
            gl.append({"tissue": t, "version": version, "kind": kind, "n_programmes": len(names), "n_rhythmic_p_perm_le_0.05": n_obs,
                       "null_mean": float(np.mean(null)), "null_q95": float(np.quantile(null, 0.95)),
                       "p_global": (1 + sum(x >= n_obs for x in null)) / len(CLASSES),
                       "rhythmic_programmes": "; ".join(n for n, q in zip(names, obs_p) if q <= 0.05)})
    g0 = [g for g in gl if g["tissue"] == t and g["version"] == "primary" and g["kind"] == "uncentred"][0]
    prim.append({"hypothesis": "M4 functional readiness", "tissue": t, "primary_statistic": "global test: number of curated function programmes rhythmic "
                 "(exact p <= 0.05) in pooled neutrophils + monocytes vs 59 joint relabelings", "primary_p": g0["p_global"], "in_window": True,
                 "detail": f"{g0['n_rhythmic_p_perm_le_0.05']} of {g0['n_programmes']} (null mean {g0['null_mean']:.2f}): {g0['rhythmic_programmes']}"})
    # ---------------- pseudobulk clock offsets (530 method)
    def clock_table(M):
        cpm = M / M.sum(0) * 1e6
        L = np.log2(cpm + 1)
        Xd = cosinor.design(zt)
        XtXi = np.linalg.inv(Xd.T @ Xd)
        rows = []
        for g in CAN:
            if g not in L.index or cpm.loc[g].mean() < 1:
                continue
            y = L.loc[g].to_numpy(float)
            b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            se = np.sqrt(((y - Xd @ b) ** 2).sum() / 3 * np.diag(XtXi))
            A_ = np.hypot(b[1], b[2])
            seA = np.sqrt((b[1] ** 2 * se[1] ** 2 + b[2] ** 2 * se[2] ** 2) / max(A_ ** 2, 1e-300))
            snr = A_ / seA if seA > 0 else 0
            if snr >= 1:
                rows.append({"gene": g, "phi": float(cosinor.amp_phase(b[1], b[2])[1]), "w": min(snr, 10) ** 2, "canon": CAN[g]})
        return pd.DataFrame(rows)

    def off_of(g, idx=None):
        g = g if idx is None else g.iloc[idx]
        dev = phase.circ_diff(g["phi"].to_numpy(), g["canon"].to_numpy()) % 24
        m, R = phase.circ_mean(dev, g["w"].to_numpy())
        return float(phase.circ_diff(m, 0)), R

    cu = {"neutrophils (pooled)": mat(NEU), "monocytes + macrophages (pooled)": mat(MONO), "myeloid (pooled)": mat([GMP] + NEU + MONO),
          "tissue pool (all cells)": mat(sorted(d0["group"].unique()))}
    tabs = {u: clock_table(M) for u, M in cu.items()}
    ref = tabs["tissue pool (all cells)"]
    for u, g in tabs.items():
        if len(g) < 2:
            continue
        o, R = off_of(g)
        bs = [off_of(g, rng.integers(0, len(g), len(g)))[0] for _ in range(2000)]
        rec = {"tissue": t, "unit": u, "n_genes": len(g), "offset_h": o, "resultant": R,
               "ci_halfwidth_h": float(np.quantile(np.abs(phase.circ_diff(np.array(bs), o)), 0.95))}
        if u != "tissue pool (all cells)":
            common = sorted(set(g["gene"]) & set(ref["gene"]))
            if len(common) >= 3:
                gu, gr = g.set_index("gene").loc[common].reset_index(), ref.set_index("gene").loc[common].reset_index()
                d_obs = float(phase.circ_diff(off_of(gu)[0], off_of(gr)[0]))
                db = np.array([float(phase.circ_diff(off_of(gu, i)[0], off_of(gr, i)[0])) for i in [rng.integers(0, len(common), len(common)) for _ in range(2000)]])
                rec.update({"diff_vs_tissue_pool_h": d_obs, "diff_ci_halfwidth_h": float(np.quantile(np.abs(phase.circ_diff(db, d_obs)), 0.95))})
        ck.append(rec)
    log(f"{t}: function and pseudobulk clocks done")

# ---------------- cell-level clock projection (332 method)
A = ad.read_h5ad(H5AD)
A.obs["tcode"] = A.obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
smp = sample_table(A.obs).set_index("sample_id")
clock = [g for g in CAN if g in A.var_names]
gi = [A.var_names.get_loc(g) for g in clock]
E = np.exp(1j * phase.W * np.array([CAN[g] for g in clock]))
Xc = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
lib = np.asarray(Xc.sum(1)).ravel()
C = Xc[:, gi].toarray()
Lc = np.log1p(C / lib[:, None] * 1e4)
STRUCT = {"Bone", "Endothelial", "Epithelial", "Glia", "Neuronal", "Sensory", "Stromal"}
cl = []
for t in ["bm", "cochlea"]:
    o = A.obs[A.obs["tcode"] == t]
    ctv = o["cell_type"].astype(str)
    groups = {"myeloid (pooled)": ctv.isin([GMP] + NEU + MONO), "neutrophils": ctv.isin(NEU), "monocytes + macrophages": ctv.isin(MONO),
              "structural cells (pooled)": o["cell_class"].astype(str).isin(STRUCT)}
    for gname, mask in groups.items():
        idx = np.where(A.obs_names.isin(o.index[mask.to_numpy()]))[0]
        cu_ = C[idx].sum(1)
        ok = idx[cu_ >= 3]
        rec = {"tissue": t, "group": gname, "n_cells": len(idx), "mean_clock_umis_per_cell": float(cu_.mean()), "frac_cells_usable": float(len(ok) / max(len(idx), 1))}
        if len(ok) >= 300:
            z = Lc[ok].copy()
            ct_ = A.obs["cell_type"].astype(str).to_numpy()[ok]
            for c_ in np.unique(ct_):
                m = ct_ == c_
                z[m] = (z[m] - z[m].mean(0)) / np.where(z[m].std(0) > 0, z[m].std(0), 1)
            v = z @ E
            d = pd.DataFrame({"sample_id": A.obs["batch_id"].astype(str).to_numpy()[ok], "phi": np.mod(np.angle(v) / phase.W, 24), "w": np.abs(v)})
            r = sorted((smp.loc[s, "zt"], *phase.circ_mean(g["phi"].to_numpy(), g["w"].to_numpy())) for s, g in d.groupby("sample_id"))
            if len(r) == 6:
                zts, mph = np.array([x[0] for x in r]), np.array([x[1] for x in r])
                off_, Roff = phase.circ_mean(phase.circ_diff(mph, zts) % 24)
                null = [phase.circ_mean(phase.circ_diff(mph, zts[np.array(k)]) % 24)[1] for k in CLASSES]
                rec.update({"n_cells_used": len(ok), "offset_resultant": Roff, "p_perm": float(np.mean(np.array(null) >= Roff - 1e-12)),
                            "mean_offset_cell_phase_minus_zt_h": float(phase.circ_diff(off_, 0)),
                            "median_within_sample_resultant": float(np.median([x[2] for x in r]))})
        cl.append(rec)
cl = pd.DataFrame(cl)
refo = cl[cl["group"] == "structural cells (pooled)"].set_index("tissue")["mean_offset_cell_phase_minus_zt_h"]
cl["offset_minus_structural_h"] = [float(phase.circ_diff(r.mean_offset_cell_phase_minus_zt_h, refo.get(r.tissue, np.nan))) if pd.notna(getattr(r, "mean_offset_cell_phase_minus_zt_h", np.nan)) else np.nan
                                   for r in cl.itertuples()]
for t in ["bm", "cochlea"]:
    r = cl[(cl["tissue"] == t) & (cl["group"] == "myeloid (pooled)")].iloc[0]
    prim.append({"hypothesis": "M7 myeloid clocks", "tissue": t, "primary_statistic": "exact permutation p of the cell-level clock tracking (offset "
                 "resultant) of pooled myeloid cells", "primary_p": r.get("p_perm", np.nan), "in_window": True,
                 "detail": f"resultant {r.get('offset_resultant', np.nan):.2f}; offset vs structural {r.get('offset_minus_structural_h', np.nan):+.1f} h; "
                           f"usable cells {r['frac_cells_usable']:.2f}"})
# ---------------- regulators
tf41 = pd.read_csv(RESULTS / "41_pathway/pathway_rhythm.csv")
tf41 = tf41[(tf41["resource"] == "collectri") & tf41["cell_type"].isin([GMP] + NEU + MONO) & tf41["pathway"].isin(TFS)].assign(version="41 uncentred")
tf43 = pd.read_csv(RESULTS / "43_pathway_phase/pathway_rhythm_centred.csv")
tf43 = tf43[(tf43["resource"] == "collectri") & tf43["cell_type"].isin([GMP] + NEU + MONO) & tf43["pathway"].isin(TFS)]
tf43 = tf43.assign(version="43 " + tf43["version"].astype(str))
lag = pd.read_csv(RESULTS / "43_pathway_phase/tf_activity_mrna_lag.csv")
lag = lag[lag["cell_type"].isin([GMP] + NEU + MONO) & lag["tf"].isin(TFS)][["tissue", "cell_type", "tf", "mrna_peak_zt", "mrna_p", "lag_h"]].rename(columns={"tf": "pathway"})
tf = pd.concat([tf41, tf43], ignore_index=True).merge(lag, on=["tissue", "cell_type", "pathway"], how="left")
tf["peak_clock"] = phase.zt_to_clock(tf["peak_zt"])

write_csv(pd.DataFrame(fr), MOD, "m_function_rhythm.csv", "Myeloid function programmes: rhythms per unit (pooled neutrophils + monocytes and each state); uncentred / centred; primary / depth-matched",
          required=["tissue", "version", "kind", "unit", "programme", "peak_zt", "peak_clock", "p_perm"])
write_csv(pd.DataFrame(gl), MOD, "m_function_global.csv", "M4 global test: rhythmic function programmes vs 59 joint relabelings", required=["tissue", "version", "kind", "p_global"])
write_csv(cl, MOD, "m_clock_cell_level.csv", "Cell-level clock projection of myeloid groups vs structural cells: tracking, offset, power", required=["tissue", "group", "n_cells"])
write_csv(pd.DataFrame(ck), MOD, "m_clock_offsets.csv", "Pseudobulk clock offsets of myeloid units with gene-bootstrap CIs and the difference vs the tissue pool", required=["tissue", "unit", "n_genes"])
write_csv(tf, MOD, "m_tf_activity.csv", "Myeloid regulators: CollecTRI activity rhythms (41 / 43) with TF activity – mRNA lags (exploratory)",
          required=["tissue", "cell_type", "pathway", "version", "peak_zt", "peak_clock", "p_perm"])
write_csv(pd.DataFrame(prim), MOD, "m_primary_function_clock.csv", "Primary hypotheses M4 and M7: primary statistics", required=["hypothesis", "tissue", "primary_p"])
log("\n" + pd.DataFrame(gl).round(3).to_string())
log("\n" + cl.round(3).to_string())
log("\n" + pd.DataFrame(ck).round(2).to_string())
log("\n" + pd.DataFrame(prim).to_string(max_colwidth=120))
