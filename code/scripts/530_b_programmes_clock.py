"""Round 5, WP-B3 — B-cell function programmes, regulators and the B-cell clock (H4; exploratory programme screen).

(1) Programmes per B stage (pseudobulk; BM and cochlea): curated `b_cell` sets (scripts/resources/curated_sets.yaml),
    Hallmark OXPHOS / glycolysis / mTORC1 / MYC targets V1 / IFN-α / IFN-γ / TNF-α–NF-κB (handling-flagged), curated
    glucocorticoid response and clock sets. Score = mean gene z (log2 CPM, z across the 6 samples, ≥ 3 expressed members)
    for every primary cell type of the tissue; 'centred' = minus the per-sample median over primary types (as 43);
    depth-matched rerun (binomial thinning to the smallest library, as 41). Cosinor, exact 60-class permutation, LOTO;
    BH within tissue × version × kind. Exploratory family (D33).
(2) Regulator activity (exploratory): CollecTRI ULM rhythms of B regulators from 41 (uncentred) and 43 (centred), and
    the TF activity – mRNA lag table of 43, restricted to B stages.
(3) Gene-level calls: B-stage Tier-1/2 (and cochlear Tier C) with driver classes (r4_drivers), stress-like and
    sample-pattern-like counts, and per-stage baboon spleen concordance (r4_crossspecies).
(4) B-cell clock (H4; D35): pseudobulk units = each B stage, the pooled B lineage, T Cell (CD3+), NK Cell, the tissue
    pool (all cells). Per unit OLS cosinor of log2 CPM for the canonical clock genes; weights w = min(A / SE(A), 10)²
    for genes with A / SE(A) ≥ 1 (230); offset = weighted circular mean of (peak − canonical phase).
    H4 primary: offset(B pooled) − offset(tissue pool) with a paired gene bootstrap (2,000; genes present in both);
    two-sided bootstrap p; determinate if the 95 % half-width ≤ 3 h. Comparators T and NK the same way.
    Cell level (agreeing method): canonical-phase projection per cell (332 method; cells with ≥ 3 clock UMIs) for the
    pooled B lineage, T, NK and the structural cells of the tissue; offset consistency over the 6 samples (resultant,
    exact permutation p) and mean offset; power = clock UMIs per cell and share of cells usable.
Outputs (results/r5_bcell/): b_programme_rhythm.csv, b_tf_activity.csv, b_gene_calls_summary.csv, b_clock_offsets.csv,
b_clock_cell_level.csv, b_primary_h4.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import CACHE, EXTERNAL, H5AD, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402

MOD = "r5_bcell"
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
COMP = ["T Cell (CD3+)", "NK Cell"]
CAN = phase.CANONICAL_CLOCK
CLASSES = cosinor.dihedral_classes(6)
rng = np.random.default_rng(31)
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
TFS = ["Ebf1", "Pax5", "Tcf3", "Ikzf1", "Ikzf3", "Foxo1", "Irf4", "Irf8", "Bach2", "Spib", "Klf2", "Mef2c", "Myc", "Spi1", "Bcl6", "Prdm1"]
HALL = ["HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_GLYCOLYSIS", "HALLMARK_MTORC1_SIGNALING", "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"]


def gmt(path):
    out = {}
    for line in open(path):
        x = line.rstrip("\n").split("\t")
        out[x[0]] = x[2:]
    return out


hm = gmt(EXTERNAL / "resources/msigdb_mh_hallmark.gmt")
GS = {f"b_cell:{k}": v for k, v in SETS["b_cell"].items()}
GS.update({f"hallmark:{k.replace('HALLMARK_', '')}": hm[k] for k in HALL if k in hm})
GS["systemic:glucocorticoid_response"] = SETS["systemic_cues"]["glucocorticoid_response"]
GS.update({f"clock:{k}": v for k, v in SETS["clock"].items()})
HANDLING_SETS = {"hallmark:TNFA_SIGNALING_VIA_NFKB", "b_cell:retention_homing", "b_cell:egress"}


def rhythm(y, zt):
    y = np.asarray(y, float)
    f = cosinor.fit(y[:, None], zt)
    obs = float(f["F"][0])
    null = [float(cosinor.fit(y[:, None], zt[list(p)])["F"][0]) for p in CLASSES[1:]]
    return {"amplitude": float(f["amp"][0]), "peak_zt": float(f["peak"][0]), "peak_clock": float(phase.zt_to_clock(f["peak"][0])),
            "p": float(f["p"][0]), "p_perm": (1 + sum(x >= obs for x in null)) / len(CLASSES),
            "loto_phase_sd_h": float(cosinor.loto(y[:, None], zt)["loto_phase_sd_h"][0])}


st_design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
prog, clk, prim = [], [], []
pb_units = {}
for t in ["bm", "cochlea"]:
    d0 = st_design[(st_design["tissue"] == t) & (st_design["level"] == "cell_type")]
    samples = d0.drop_duplicates("sample_id").sort_values("zt")["sample_id"].tolist()
    zt = d0.drop_duplicates("sample_id").sort_values("zt")["zt"].to_numpy(float)
    prim_types = sorted(d0.loc[d0["status"] == "primary", "group"].unique())
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)

    def mat(types):
        return sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples))

    units = {ty: mat([ty]) for ty in prim_types}
    units["B lineage (pooled)"] = mat(BST)
    all_types = sorted(d0["group"].unique())
    units["tissue pool (all cells)"] = mat(all_types)
    pb_units[t] = units
    depth = {u: M.sum(0).to_numpy(float) for u, M in units.items()}

    def thin(M):
        k = M.sum(0).min()
        return M.apply(lambda c: pd.Series(rng.binomial(c.astype(int).to_numpy(), k / c.sum()), index=c.index))

    # ---------------- (1) programmes
    for version, U in [("primary", units), ("depth_matched", {u: thin(M) for u, M in units.items() if u in prim_types})]:
        scores = {}
        for u, M in U.items():
            if u not in prim_types:
                continue
            cpm = M / M.sum(0) * 1e6
            L = np.log2(cpm + 1)[cpm.mean(1) >= 1]
            Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
            for gs, genes in GS.items():
                g = [x for x in genes if x in Z.index]
                if len(g) >= 3:
                    scores[(u, gs)] = (Z.loc[g].mean(0).to_numpy(float), len(g))
        for gs in GS:
            vals = [scores[(u, gs)][0] for u in prim_types if (u, gs) in scores]
            if not vals:
                continue
            med = np.nanmedian(np.vstack(vals), axis=0)
            for u in [b for b in BST if b in prim_types]:
                if (u, gs) not in scores:
                    continue
                y0, ng = scores[(u, gs)]
                for kind, y in [("uncentred", y0), ("centred", y0 - med)]:
                    if np.isnan(y).any():
                        continue
                    r = rhythm(y, zt)
                    prog.append({"tissue": t, "version": version, "kind": kind, "stage": u, "gene_set": gs, "n_genes": ng,
                                 "handling_flag": gs in HANDLING_SETS, **r,
                                 "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1]),
                                 "r_with_library_size": float(np.corrcoef(y, depth[u])[0, 1])})
    # ---------------- (4) clock offsets (pseudobulk)
    def clock_table(M):
        cpm = M / M.sum(0) * 1e6
        L = np.log2(cpm + 1)
        X = cosinor.design(zt)
        XtXi = np.linalg.inv(X.T @ X)
        rows = []
        for g in CAN:
            if g not in L.index or cpm.loc[g].mean() < 1:
                continue
            y = L.loc[g].to_numpy(float)
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            s2 = ((y - X @ b) ** 2).sum() / 3
            se = np.sqrt(s2 * np.diag(XtXi))
            A = np.hypot(b[1], b[2])
            seA = np.sqrt((b[1] ** 2 * se[1] ** 2 + b[2] ** 2 * se[2] ** 2) / max(A ** 2, 1e-300))
            snr = A / seA if seA > 0 else 0
            if snr < 1:
                continue
            rows.append({"gene": g, "phi": float(cosinor.amp_phase(b[1], b[2])[1]), "w": min(snr, 10) ** 2, "canon": CAN[g]})
        return pd.DataFrame(rows)

    tabs = {u: clock_table(M) for u, M in units.items() if u in BST + COMP + ["B lineage (pooled)", "tissue pool (all cells)"]}

    def off_of(g, idx=None):
        g = g if idx is None else g.iloc[idx]
        dev = phase.circ_diff(g["phi"].to_numpy(), g["canon"].to_numpy()) % 24
        m, R = phase.circ_mean(dev, g["w"].to_numpy())
        return float(phase.circ_diff(m, 0)), R

    ref = tabs["tissue pool (all cells)"]
    for u, g in tabs.items():
        if len(g) < 2:
            clk.append({"tissue": t, "unit": u, "n_genes": len(g)})
            continue
        o, R = off_of(g)
        bs = [off_of(g, rng.integers(0, len(g), len(g)))[0] for _ in range(2000)]
        half = float(np.quantile(np.abs(phase.circ_diff(np.array(bs), o)), 0.95))
        rec = {"tissue": t, "unit": u, "n_genes": len(g), "genes": ";".join(g["gene"]), "offset_h": o, "resultant": R,
               "ci_halfwidth_h": half, "determinate": bool(half <= 3 and len(g) >= 4)}
        if u != "tissue pool (all cells)":
            common = sorted(set(g["gene"]) & set(ref["gene"]))
            if len(common) >= 3:
                gu, gr = g.set_index("gene").loc[common].reset_index(), ref.set_index("gene").loc[common].reset_index()
                d_obs = float(phase.circ_diff(off_of(gu)[0], off_of(gr)[0]))
                db = []
                for _ in range(2000):
                    i = rng.integers(0, len(common), len(common))
                    db.append(float(phase.circ_diff(off_of(gu, i)[0], off_of(gr, i)[0])))
                db = np.array(db)
                dhalf = float(np.quantile(np.abs(phase.circ_diff(db, d_obs)), 0.95))
                pdiff = float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean())))
                rec.update({"n_common_genes": len(common), "diff_vs_tissue_pool_h": d_obs, "diff_ci_halfwidth_h": dhalf,
                            "diff_p_bootstrap": pdiff, "diff_determinate": bool(dhalf <= 3)})
        clk.append(rec)
    log(f"{t}: programmes and clock offsets done")

# ---------------- cell-level clock projection (agreeing method for H4)
A = ad.read_h5ad(H5AD)
A.obs["tcode"] = A.obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
smp = sample_table(A.obs).set_index("sample_id")
clock = [g for g in CAN if g in A.var_names]
gi = [A.var_names.get_loc(g) for g in clock]
E = np.exp(1j * phase.W * np.array([CAN[g] for g in clock]))
X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
lib = np.asarray(X.sum(1)).ravel()
C = X[:, gi].toarray()
Lc = np.log1p(C / lib[:, None] * 1e4)
STRUCT_CLASSES = {"Bone", "Endothelial", "Epithelial", "Glia", "Neuronal", "Sensory", "Stromal"}
cell_rows = []
for t in ["bm", "cochlea"]:
    obs = A.obs[A.obs["tcode"] == t]
    groups = {"B lineage (pooled)": obs["cell_type"].astype(str).isin(BST)}
    for cty in COMP:
        groups[cty] = obs["cell_type"].astype(str) == cty
    groups["structural cells (pooled)"] = obs["cell_class"].astype(str).isin(STRUCT_CLASSES)
    for gname, mask in groups.items():
        idx = np.where(A.obs_names.isin(obs.index[mask.to_numpy()]))[0]
        if len(idx) == 0:
            continue
        clock_umis = C[idx].sum(1)
        ok = idx[clock_umis >= 3]
        rec = {"tissue": t, "group": gname, "n_cells": len(idx), "mean_clock_umis_per_cell": float(clock_umis.mean()),
               "frac_cells_usable": float(len(ok) / len(idx))}
        if len(ok) >= 300:
            z = Lc[ok]
            # z within cell type (as 332): centre each gene within each gold-standard type of the group
            ct = A.obs["cell_type"].astype(str).to_numpy()[ok]
            for c_ in np.unique(ct):
                m = ct == c_
                z[m] = (z[m] - z[m].mean(0)) / np.where(z[m].std(0) > 0, z[m].std(0), 1)
            v = z @ E
            d = pd.DataFrame({"sample_id": A.obs["batch_id"].astype(str).to_numpy()[ok], "phi": np.mod(np.angle(v) / phase.W, 24), "w": np.abs(v)})
            r = []
            for sid, g in d.groupby("sample_id"):
                m_, R_ = phase.circ_mean(g["phi"].to_numpy(), g["w"].to_numpy())
                r.append((smp.loc[sid, "zt"], m_, R_))
            r = sorted(r)
            zts, mph = np.array([x[0] for x in r]), np.array([x[1] for x in r])
            if len(r) == 6:
                off_, Roff = phase.circ_mean(phase.circ_diff(mph, zts) % 24)
                null = [phase.circ_mean(phase.circ_diff(mph, zts[np.array(k)]) % 24)[1] for k in CLASSES]
                rec.update({"n_cells_used": len(ok), "offset_resultant": Roff, "p_perm": float(np.mean(np.array(null) >= Roff - 1e-12)),
                            "mean_offset_cell_phase_minus_zt_h": float(phase.circ_diff(off_, 0)),
                            "median_within_sample_resultant": float(np.median([x[2] for x in r]))})
        cell_rows.append(rec)
cl = pd.DataFrame(cell_rows)
ref_off = cl[cl["group"] == "structural cells (pooled)"].set_index("tissue")["mean_offset_cell_phase_minus_zt_h"]
cl["offset_minus_structural_h"] = [float(phase.circ_diff(r.mean_offset_cell_phase_minus_zt_h, ref_off.get(r.tissue, np.nan)))
                                   if pd.notna(r.mean_offset_cell_phase_minus_zt_h) else np.nan for r in cl.itertuples()] \
    if "mean_offset_cell_phase_minus_zt_h" in cl else np.nan

ck = pd.DataFrame(clk)
for t in ["bm", "cochlea"]:
    r = ck[(ck["tissue"] == t) & (ck["unit"] == "B lineage (pooled)")].iloc[0]
    c = cl[(cl["tissue"] == t) & (cl["group"] == "B lineage (pooled)")]
    prim.append({"hypothesis": "H4 B-clock shift", "tissue": t,
                 "primary_statistic": "paired gene-bootstrap p of the pooled B-lineage clock offset minus the tissue-pool offset",
                 "primary_p": r.get("diff_p_bootstrap", np.nan), "shift_h": r.get("diff_vs_tissue_pool_h", np.nan),
                 "ci_halfwidth_h": r.get("diff_ci_halfwidth_h", np.nan), "determinate": bool(r.get("diff_determinate", False)),
                 "detail": (f"B offset {r['offset_h']:+.2f} h ({int(r['n_genes'])} genes); cell level: offset vs structural "
                            f"{c['offset_minus_structural_h'].iloc[0]:+.2f} h, tracking p {c['p_perm'].iloc[0]:.3f}") if len(c) and "p_perm" in c else ""})

pr = pd.DataFrame(prog)
for (t, v, k), idx in pr.groupby(["tissue", "version", "kind"]).groups.items():
    pr.loc[idx, "q_BH"] = multipletests(pr.loc[idx, "p"], method="fdr_bh")[1]
pr["robust"] = (pr["q_BH"] < 0.1) & (pr["p_perm"] <= 0.05) & (pr["loto_phase_sd_h"] < 3)
write_csv(pr, MOD, "b_programme_rhythm.csv", "B-stage function programmes: rhythms (pseudobulk; uncentred / centred; primary / depth-matched); exploratory",
          required=["tissue", "version", "kind", "stage", "gene_set", "peak_zt", "peak_clock", "p_perm", "q_BH"])
# ---------------- (2) regulators
tf41 = pd.read_csv(RESULTS / "41_pathway/pathway_rhythm.csv")
tf41 = tf41[(tf41["resource"] == "collectri") & tf41["cell_type"].isin(BST) & tf41["pathway"].isin(TFS)].assign(version="41 uncentred")
tf43 = pd.read_csv(RESULTS / "43_pathway_phase/pathway_rhythm_centred.csv")
tf43 = tf43[(tf43["resource"] == "collectri") & tf43["cell_type"].isin(BST) & tf43["pathway"].isin(TFS)]
tf43 = tf43.assign(version="43 " + tf43["version"].astype(str))
lag = pd.read_csv(RESULTS / "43_pathway_phase/tf_activity_mrna_lag.csv")
lag = lag[lag["cell_type"].isin(BST) & lag["tf"].isin(TFS)][["tissue", "cell_type", "tf", "mrna_peak_zt", "mrna_p", "lag_h"]].rename(columns={"tf": "pathway"})
tf = pd.concat([tf41, tf43], ignore_index=True).merge(lag, on=["tissue", "cell_type", "pathway"], how="left")
tf["peak_clock"] = phase.zt_to_clock(tf["peak_zt"])
write_csv(tf, MOD, "b_tf_activity.csv", "B regulators: CollecTRI activity rhythms per B stage (41 uncentred; 43 centred) with TF-activity – mRNA lag",
          required=["tissue", "cell_type", "pathway", "version", "peak_zt", "peak_clock", "p_perm"])
# ---------------- (3) gene-level calls
dc = pd.read_csv(RESULTS / "r4_drivers/driver_calls.csv")
dc = dc[dc["cell_type"].isin(BST)]
gsum = dc.groupby(["tissue", "cell_type", "tier"]).agg(
    n_calls=("gene", "size"), n_handling_like=("handling_linked", "sum"), n_pattern_like=("pattern_linked", "sum"),
    n_clock_controlled=("clock_controlled", "sum"), n_composition_linked=("composition_linked", "sum"),
    circ_mean_peak_zt=("peak_zt", lambda x: phase.circ_mean(x.to_numpy())[0])).reset_index()
gsum["n_calls_without_handling_or_pattern"] = gsum["n_calls"] - dc.groupby(["tissue", "cell_type", "tier"]).apply(
    lambda g: int((g["handling_linked"] | g["pattern_linked"]).sum())).to_numpy()
bc = pd.read_csv(RESULTS / "r4_crossspecies/baboon_bm_gene_concordance.csv")
bc = bc[bc["our_scope"].isin(BST) & (bc["genes"] == "non_clock")][["baboon_tissue", "our_scope", "n_genes", "frac_within_3h", "null_frac_within_3h", "p_perm"]]
bc = bc.pivot_table(index="our_scope", columns="baboon_tissue", values=["frac_within_3h", "null_frac_within_3h", "p_perm"])
bc.columns = [f"baboon_{b}_{a}" for a, b in bc.columns]
gsum = gsum.merge(bc.reset_index().rename(columns={"our_scope": "cell_type"}).assign(tissue="bm"), on=["tissue", "cell_type"], how="left")
write_csv(gsum, MOD, "b_gene_calls_summary.csv", "B-stage gene-level calls with driver / handling / pattern counts and baboon spleen / BM concordance",
          required=["tissue", "cell_type", "tier", "n_calls"])
write_csv(ck, MOD, "b_clock_offsets.csv", "Clock offsets (pseudobulk, gene bootstrap) of B stages, pooled B lineage, T, NK and the tissue pool; difference vs tissue pool",
          required=["tissue", "unit", "n_genes"])
write_csv(cl, MOD, "b_clock_cell_level.csv", "Cell-level clock projection: pooled B lineage, T, NK and structural cells — tracking of ZT, offset, power",
          required=["tissue", "group", "n_cells", "mean_clock_umis_per_cell"])
write_csv(pd.DataFrame(prim), MOD, "b_primary_h4.csv", "Primary hypothesis H4 (B-clock shift): primary statistic per tissue",
          required=["hypothesis", "tissue", "primary_p", "determinate"])
log("\n" + ck.round(3).to_string())
log("\n" + cl.round(3).to_string())
log("\n" + pr[(pr["version"] == "primary") & (pr["kind"] == "centred") & (pr["p_perm"] <= 0.05)][["tissue", "stage", "gene_set", "peak_zt", "p_perm", "q_BH", "robust"]].round(3).to_string())
