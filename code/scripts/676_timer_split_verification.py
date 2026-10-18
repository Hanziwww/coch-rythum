"""Round 6 follow-up — verification of the neutrophil-timer split (G02) vs the 'sample-level artefact' reading (675).

675 found that within-sample centring removes the BM negative sign. Centring cannot distinguish an artefact from a
genuine marrow-wide programme, and it is invalid if the reference cell types carry neutrophil-derived signal. Tests:
(A) Exact time-relabelling null (60 dihedral classes; sample = unit) for the G02 statistic (Spearman of the predicted
    ZT13 − ZT5 difference with the GSE86619 blood change) in every primary cell type of each tissue — the gene-label
    permutation p of 77 ignores that one sample-level profile drives thousands of genes.
(B) Specificity and its source: the same statistic in every primary cell type, overall and split into genes enriched in
    pooled neutrophils (log2 CPM >= 2 above the median of the tissue's non-myeloid types, mean CPM >= 5 in neutrophils)
    vs non-enriched genes (|difference| < 1). A negative sign in non-myeloid marrow types carried by neutrophil-enriched
    genes indicates neutrophil-derived signal in those types (which would make 675's centring remove real signal).
(C) Centring restricted to non-enriched genes (reference profiles cannot carry neutrophil-specific transcripts).
(D) External replication under the same exact null: GSE198770 long-bone BM (control; neutrophil-depleted), cosinor on
    log2 normalised counts (18 samples; clock-aligned ZT as 246), genes rhythmic at p < 0.05 and all genes; exact p over
    the 60 relabelings of the six ZT groups (replicates kept together; rhythmic genes re-selected per relabeling).
Output (results/r6_myeloid/): m_timer_split_verification.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r6_myeloid"
W = 2 * np.pi / 24
DC, DS = np.cos(13 * W) - np.cos(5 * W), np.sin(13 * W) - np.sin(5 * W)
CLS = cosinor.dihedral_classes(6)
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MYE = [GMP] + NEU + ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage", "Mast Cell"]
obs = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_gene_table.csv").set_index("gene")["wt_zt13_vs_zt5_log2fc"].dropna()
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")


def lcpm(pb, types, samples):
    ok = [ty for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples)]
    M = sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in ok)
    cpm = M / M.sum(0) * 1e6
    return np.log2(cpm + 1), cpm.mean(1)


def rho_of(Y, zt, ob):
    f = cosinor.fit(Y.T, zt)
    return stats.spearmanr(f["a"] * DC + f["b"] * DS, ob)[0]


def exact(Y, zt, ob):
    """observed rho and its exact two-sided / one-sided ranks over the 60 relabelings (identity first)"""
    r = np.array([rho_of(Y, zt[list(c)], ob) for c in CLS])
    return r[0], float(np.mean(np.abs(r) >= abs(r[0]) - 1e-12)), float(np.mean(r <= r[0] + 1e-12)), float(np.mean(r >= r[0] - 1e-12))


rows = []
for t in ["bm", "cochlea"]:
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    s0 = d0.drop_duplicates("sample_id").sort_values("zt")
    samples, zt = s0["sample_id"].tolist(), s0["zt"].to_numpy(float)
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    prim = sorted(d0.loc[d0["status"] == "primary", "group"].unique())
    refs = [ty for ty in prim if ty not in MYE]
    Ln, mn = lcpm(pb, NEU, samples)
    Lr = {ty: lcpm(pb, [ty], samples) for ty in refs}
    ref_mean = pd.concat([L.mean(1) for L, _ in Lr.values()], axis=1).median(1)
    enr = Ln.mean(1) - ref_mean
    neu_enriched = set(enr.index[(enr >= 2) & (mn >= 5)])
    shared = set(enr.index[enr.abs() < 1])
    units = [("pooled neutrophils", NEU)] + [(ty, [ty]) for ty in prim]
    for unit, types in units:
        L, m = lcpm(pb, types, samples)
        base = [g for g in L.index[(m >= 5).to_numpy()] if g in obs.index]
        for gs, genes in [("all shared genes", base), ("neutrophil-enriched genes", [g for g in base if g in neu_enriched]),
                          ("non-enriched genes", [g for g in base if g in shared])]:
            if len(genes) < 30:
                continue
            Y = L.loc[genes].to_numpy(float)
            r0, p2, p_lo, p_hi = exact(Y, zt, obs[genes].to_numpy())
            rows.append({"part": "A/B our data (raw, as G02)", "tissue": t, "unit": unit, "unit_class": "neutrophil lineage" if unit in NEU + ["pooled neutrophils"]
                         else ("other myeloid" if unit in MYE else "non-myeloid"), "gene_set": gs, "n_genes": len(genes), "spearman": r0,
                         "p_exact_two_sided": p2, "p_exact_negative": p_lo, "p_exact_positive": p_hi})
    # (C) centring restricted to non-enriched genes
    stack = []
    for ty in refs:
        L, m = Lr[ty]
        D = L.sub(L.mean(1), axis=0)
        D[(m < 5).to_numpy()] = np.nan
        stack.append(D)
    centre = pd.concat(stack, axis=0, keys=range(len(stack))).groupby(level=1).median()
    nref = pd.concat([d.iloc[:, 0].notna() for d in stack], axis=1).sum(1)
    for unit, types in [("pooled neutrophils", NEU)] + [(ty, [ty]) for ty in NEU]:
        L, m = lcpm(pb, types, samples)
        D = L.sub(L.mean(1), axis=0)
        for gs, pool in [("all shared genes", None), ("non-enriched genes", shared)]:
            genes = [g for g in D.index[(m >= 5).to_numpy()] if g in obs.index and g in centre.index and nref.get(g, 0) >= 3
                     and np.isfinite(centre.loc[g]).all() and (pool is None or g in pool)]
            Y = (D.loc[genes] - centre.loc[genes]).to_numpy(float)
            r0, p2, p_lo, p_hi = exact(Y, zt, obs[genes].to_numpy())
            rows.append({"part": "C our data (within-sample centred)", "tissue": t, "unit": unit, "unit_class": "neutrophil lineage", "gene_set": gs,
                         "n_genes": len(genes), "spearman": r0, "p_exact_two_sided": p2, "p_exact_negative": p_lo, "p_exact_positive": p_hi})
    log(f"{t}: {len(neu_enriched)} neutrophil-enriched, {len(shared)} non-enriched genes; {len(refs)} non-myeloid reference types")

# (D) external long-bone BM
meta = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__meta.csv")
ex = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__expr.csv", index_col=0)
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
for cond in ["control", "neutrophil_depleted"]:
    m = meta[meta["condition"] == cond]
    L = np.log2(ex[m["sample"]].astype(float) + 1)
    L = L[(L.mean(1) >= 1) & L.index.isin(obs.index)]
    o_ = off[(off["dataset"] == "GSE198770_bone_marrow") & (off["condition"] == cond)]["offset_canonical_h"]
    o_ = float(o_.iloc[0]) if len(o_) else 0.0
    tt = (m["time_h"].to_numpy(float) - o_) % 24
    groups = np.unique(m["time_h"].to_numpy(float))
    gidx = [np.where(m["time_h"].to_numpy(float) == g)[0] for g in groups]
    ob = obs[L.index].to_numpy()
    for gs in ["all genes", "rhythmic genes (p < 0.05)"]:
        rr = []
        for c in CLS:
            tz = np.empty(len(tt))
            for i, gi in enumerate(gidx):
                tz[gi] = (groups[c[i]] - o_) % 24
            f = cosinor.fit(L.T.to_numpy(), tz)
            sel = np.ones(len(L), bool) if gs == "all genes" else (f["p"] < 0.05)
            rr.append(stats.spearmanr((f["a"] * DC + f["b"] * DS)[sel], ob[sel])[0])
        rr = np.array(rr)
        rows.append({"part": "D external GSE198770 long-bone BM", "tissue": "bm (external)", "unit": f"whole marrow, {cond}", "unit_class": "bulk",
                     "gene_set": gs, "n_genes": int(len(L) if gs == "all genes" else (cosinor.fit(L.T.to_numpy(), tt)["p"] < 0.05).sum()),
                     "spearman": rr[0], "p_exact_two_sided": float(np.mean(np.abs(rr) >= abs(rr[0]) - 1e-12)),
                     "p_exact_negative": float(np.mean(rr <= rr[0] + 1e-12)), "p_exact_positive": float(np.mean(rr >= rr[0] - 1e-12))})
res = pd.DataFrame(rows)
write_csv(res, MOD, "m_timer_split_verification.csv", "Verification of the neutrophil-timer split: exact time-relabelling p, cell-type specificity, "
          "neutrophil-enriched vs non-enriched genes, centring on non-enriched genes, external long-bone BM with and without neutrophils",
          required=["part", "tissue", "unit", "gene_set", "spearman", "p_exact_two_sided"])
pd.set_option("display.width", 250)
log("\n" + res.round(3).to_string(index=False))
