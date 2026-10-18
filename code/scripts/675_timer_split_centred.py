"""Round 6 follow-up — is the neutrophil-timer split (G02: BM states predict the blood ZT13 − ZT5 change negatively,
cochlear states positively) neutrophil-specific or carried by the tissue-wide sample-level pattern?

640 showed that the BM blood-timer score loses its light-phase peak after within-sample centring. This repeats G02's
gene-level test (77 method) on centred profiles.
Per tissue × neutrophil-lineage state (four labels, GMP, pooled neutrophils): pseudobulk log2 CPM (prior 1; genes with mean
CPM >= 5 in the state and present in the GSE86619 table), each gene mean-centred over the 6 samples.
  raw      = the state's centred profile;
  centred  = raw minus, per gene and sample, the median of the same mean-centred profile over the tissue's primary
             non-myeloid cell types expressing the gene (mean CPM >= 5; >= 3 types) — removes shifts shared by all cell
             types of a sample (sample-level pattern, depth), as 43 / 640 for programmes.
Cosinor fit per gene → predicted ZT13 − ZT5 difference Δ̂ = a(cos 13ω − cos 5ω) + b(sin 13ω − sin 5ω); Spearman with the
observed GSE86619 WT log2FC (all shared genes; and without the 77 stress / IEG genes); gene-label permutation p (2,000,
two-sided). Also: Spearman of each state's raw Δ̂ with the Δ̂ implied by the sample-pattern PC1 profile (210) projected
onto genes (how much of the raw prediction the pattern explains).
Output (results/r6_myeloid/): m_timer_split_centred.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.dr import STRESS  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r6_myeloid"
rng = np.random.default_rng(675)
W = 2 * np.pi / 24
DC, DS = np.cos(13 * W) - np.cos(5 * W), np.sin(13 * W) - np.sin(5 * W)
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MYE = [GMP] + NEU + ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage", "Mast Cell"]
obs = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_gene_table.csv").set_index("gene")["wt_zt13_vs_zt5_log2fc"].dropna()
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")


def prof(pb, types, samples):
    ok = [ty for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples)]
    M = sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in ok)
    cpm = M / M.sum(0) * 1e6
    L = np.log2(cpm + 1)
    return L.sub(L.mean(1), axis=0), cpm.mean(1)


def delta(Y, zt):
    f = cosinor.fit(Y.T.to_numpy(), zt)
    return pd.Series(f["a"] * DC + f["b"] * DS, index=Y.index)


def sp_perm(x, y, n=2000):
    r = stats.spearmanr(x, y)[0]
    yr = stats.rankdata(y)
    xr = stats.rankdata(x)
    null = np.array([np.corrcoef(xr, rng.permutation(yr))[0, 1] for _ in range(n)])
    return float(r), float((1 + (np.abs(null) >= abs(r)).sum()) / (n + 1))


rows = []
for t in ["bm", "cochlea"]:
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    s0 = d0.drop_duplicates("sample_id").sort_values("zt")
    samples, zt = s0["sample_id"].tolist(), s0["zt"].to_numpy(float)
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    refs = [ty for ty in sorted(d0.loc[d0["status"] == "primary", "group"].unique()) if ty not in MYE]
    R = {ty: prof(pb, [ty], samples) for ty in refs}
    genes_all = pb.index
    stack = np.full((len(refs), len(genes_all), len(samples)), np.nan)
    for i, ty in enumerate(refs):
        D, m = R[ty]
        D = D.reindex(genes_all)
        keep = (m.reindex(genes_all) >= 5).to_numpy()
        stack[i, keep] = D.to_numpy()[keep]
    n_ref = (~np.isnan(stack[:, :, 0])).sum(0)
    centre = pd.DataFrame(np.nanmedian(stack, axis=0), index=genes_all, columns=samples)
    centre[n_ref < 3] = np.nan
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    for unit, types in [("pooled neutrophils", NEU)] + [(ty, [ty]) for ty in [GMP] + NEU]:
        D, m = prof(pb, types, samples)
        g = [x for x in D.index[(m >= 5).to_numpy()] if x in obs.index and np.isfinite(centre.loc[x]).all()]
        raw, cen = D.loc[g], D.loc[g] - centre.loc[g]
        # pattern-implied prediction: each gene's projection on the PC1 sample profile
        z = (pc1 - pc1.mean()) / pc1.std()
        beta = raw.to_numpy() @ z / (z @ z)
        patt = delta(pd.DataFrame(np.outer(beta, z), index=g, columns=samples), zt)
        for version, Y in [("raw (as G02)", raw), ("within-sample centred", cen)]:
            dh = delta(Y, zt)
            for gs, sel in [("all shared genes", np.ones(len(g), bool)), ("without stress / IEG genes", ~pd.Index(g).isin(STRESS))]:
                r, p = sp_perm(dh[sel].to_numpy(), obs[np.array(g)[sel]].to_numpy())
                rows.append({"tissue": t, "state": unit, "version": version, "gene_set": gs, "n_genes": int(sel.sum()), "spearman_vs_gse86619": r,
                             "p_perm": p, "spearman_with_pattern_implied_prediction": float(stats.spearmanr(dh[sel], patt[sel])[0]),
                             "n_reference_types": len(refs)})
    log(f"{t}: done ({len(refs)} reference types)")
res = pd.DataFrame(rows)
write_csv(res, MOD, "m_timer_split_centred.csv", "Neutrophil-timer split (G02) with and without within-sample centring: Spearman of predicted ZT13 − ZT5 "
          "with the GSE86619 blood change, gene-label permutation p; overlap with the sample-pattern-implied prediction",
          required=["tissue", "state", "version", "gene_set", "spearman_vs_gse86619", "p_perm"])
log("\n" + res[res["gene_set"] == "all shared genes"].round(3).to_string(index=False))
