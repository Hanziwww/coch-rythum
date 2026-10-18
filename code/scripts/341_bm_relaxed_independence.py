"""Round 3, WP5 — does the bone-marrow rhythm evidence predict held-out external data?

Held-out validation data: GSE198770 BM control replicate 3 (6 time points; replicates 1–2 were the external support
used by the BM relaxed tier, 211 with TISSUE=bm). Per gene, observed profile = centred log2(x + 1) of replicate 3;
prediction = our cosinor (a, b on our ZT) evaluated at the clock-aligned replicate-3 times (t − offset_ext +
offset_BM), averaged over the cell types in which the gene belongs to the set. Statistics over all genes × time points
of a set: Pearson r (stacked, centred) with a gene-label permutation p (2,000), and the partial r of our prediction
given the replicate-1–2 external prediction (does our fit add information beyond the external support?).
Sets: selection-free (BM Tier 1; BM tissue pseudobulk p < 0.05; Tier 1 split by tissue-pattern-likeness from 210),
relaxed-tier sets (selected rule; cochlea-support-only = no external information; external-supported; R0 outside the
selected rule; tested but p >= 0.05).
Output (results/r3_bm/): bm_independence_validation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r3_bm"
W = phase.W
rng = np.random.default_rng(341)
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
D_BM = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "bm") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
D_EXT = float(off[(off["dataset"] == "GSE198770_bone_marrow") & (off["condition"] == "control") & (off["unit"] == "bulk")]["offset_canonical_h"].iloc[0])

meta = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__meta.csv")
meta = meta[meta["condition"] == "control"]
ex = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__expr.csv", index_col=0)
ex = ex[~ex.index.duplicated()]
L = np.log2(ex[meta["sample"]] + 1)
L = L[L.mean(1) > 1]
m3 = meta[meta["replicate"] == 3].sort_values("time_h")
m12 = meta[meta["replicate"].isin([1, 2])]
t3 = m3["time_h"].to_numpy(float)
Y3 = L[m3["sample"]].to_numpy()
Y3 = Y3 - Y3.mean(1, keepdims=True)
obs = pd.DataFrame(Y3, index=L.index)
# external prediction from replicates 1–2 (OLS cosinor per gene)
t12 = m12["time_h"].to_numpy(float)
X12 = np.column_stack([np.ones_like(t12), np.cos(W * t12), np.sin(W * t12)])
B = np.linalg.lstsq(X12, L[m12["sample"]].to_numpy().T, rcond=None)[0]
Xp = np.column_stack([np.cos(W * t3), np.sin(W * t3)])
ext_pred = pd.DataFrame((Xp @ B[1:]).T, index=L.index)
ext_pred = ext_pred.sub(ext_pred.mean(1), axis=0)
tau = t3 - D_EXT + D_BM                                  # replicate-3 times on our ZT axis
Xo = np.column_stack([np.cos(W * tau), np.sin(W * tau)])


def predict(tab):
    """Mean predicted profile per gene over the rows (cell types) of a set."""
    tab = tab[tab["gene"].isin(obs.index)]
    P = tab[["a", "b"]].to_numpy() @ Xo.T
    P = P - P.mean(1, keepdims=True)
    return pd.DataFrame(P, index=tab["gene"].to_numpy()).groupby(level=0).mean()


def evaluate(name, tab, selection):
    pr = predict(tab)
    if len(pr) < 10:
        return {"gene_set": name, "selection": selection, "n_genes": len(pr)}
    o = obs.loc[pr.index].to_numpy()
    e = ext_pred.loc[pr.index].to_numpy()
    p = pr.to_numpy()
    r = float(np.corrcoef(p.ravel(), o.ravel())[0, 1])
    null = np.array([np.corrcoef(p[rng.permutation(len(p))].ravel(), o.ravel())[0, 1] for _ in range(2000)])
    z = np.column_stack([np.ones(e.size), e.ravel()])
    res = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
    per_gene = np.array([np.corrcoef(p[i], o[i])[0, 1] if p[i].std() > 0 and o[i].std() > 0 else np.nan for i in range(len(p))])
    return {"gene_set": name, "selection": selection, "n_genes": len(pr), "pearson_r_stacked": r,
            "p_perm": float((1 + (null >= r).sum()) / 2001),
            "pearson_r_external_rep12_prediction": float(np.corrcoef(e.ravel(), o.ravel())[0, 1]),
            "partial_r_ours_given_external": float(np.corrcoef(res(p.ravel()), res(o.ravel()))[0, 1]),
            "frac_genes_positive_profile_r": float(np.nanmean(per_gene > 0))}


rows = []
t1 = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier", "a", "b"])
t1 = t1[t1["tier"] == "tier1"]
pat = pd.read_csv(RESULTS / "r2_cochlea/gene_pattern_scores_bm.csv").rename(columns={"cell_type": "group"})
t1 = t1.merge(pat[["group", "gene", "pattern_r"]], on=["group", "gene"], how="left")
rows.append(evaluate("BM Tier 1", t1, "selection-free (internal evidence only)"))
rows.append(evaluate("BM Tier 1, tissue-pattern-like (r > 0.8)", t1[t1["pattern_r"] > 0.8], "selection-free"))
rows.append(evaluate("BM Tier 1, not pattern-like", t1[~(t1["pattern_r"] > 0.8)], "selection-free"))
tp = pd.read_csv(RESULTS / "r3_bm/bm_tissue_pseudobulk_cosinor.csv")
rows.append(evaluate("BM tissue pseudobulk p < 0.05", tp[tp["p"] < 0.05], "selection-free"))
rt = pd.read_csv(RESULTS / "r3_bm/bm_relaxed_tier.csv",
                 usecols=["cell_type", "gene", "a", "b", "p", "tierC", "tierC_prespecified", "s_external", "s_cochlea", "rule_selected"])
for c in ["tierC", "tierC_prespecified", "s_external", "s_cochlea"]:
    rt[c] = rt[c].astype(str) == "True"
sel = rt["rule_selected"].iloc[0]
rows.append(evaluate(f"BM relaxed tier (rule {sel})", rt[rt["tierC"]], "uses external replicates 1–2 and/or cochlea"))
rows.append(evaluate("BM relaxed, cochlea support only", rt[rt["tierC"] & rt["s_cochlea"] & ~rt["s_external"]],
                     "no external information"))
rows.append(evaluate("BM relaxed, external support (replicates 1–2)", rt[rt["tierC"] & rt["s_external"]],
                     "external replicates 1–2 (replicate 3 held out)"))
rows.append(evaluate("BM R0 calls outside the selected rule", rt[rt["tierC_prespecified"] & ~rt["tierC"]], "edgeR support only"))
rows.append(evaluate("BM tested, p >= 0.05", rt[rt["p"] >= 0.05], "background"))
res = pd.DataFrame(rows)
res["pattern_like_share_of_tier1"] = float((t1["pattern_r"] > 0.8).mean())
write_csv(res, MOD, "bm_independence_validation.csv",
          "BM rhythm sets vs held-out GSE198770 BM replicate 3: prediction r, permutation p, partial r given the external "
          "replicate-1–2 prediction", required=["gene_set", "n_genes", "pearson_r_stacked", "p_perm"])
log("\n" + res.round(3).to_string())
