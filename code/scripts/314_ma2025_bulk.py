"""Round 3, WP2 — Ma 2025 bulk cochlea (CRA019827; quantified and uploaded by the user): control vs noise × day vs night.

Usage: python scripts/314_ma2025_bulk.py [ma2025 | ma2025dry]
Samples CMD1–3, CMN1–3, NMD1–3, NMN1–3: C = before noise, N = 1 day after noise; D = day group (exposure ZT2–4),
N(suffix) = night group (exposure ZT14–16) (Ma 2025 methods). Input: a gene × sample count matrix (csv / tsv /
txt, optionally gzipped; first column gene symbol or Ensembl ID) whose name contains CRA019827 or 'bulk', under
external/ma2025, external/ma2025 or $MA_ROOT (dry run).
(a) log2 CPM (genes > 1 CPM in >= 3 samples), per-gene OLS on the four group means → contrasts: baseline night − day
    (CMN − CMD), noise effect in the day / night group, noise × time interaction (t tests, BH).
(b) Sampling phase per sample from core-clock genes (template matching with GSE88954 sham fits and with our cochlear
    tissue-pseudobulk fits, as in 252) → are the day and night groups collected at different times of day?
(c) If so, our cochlear rhythms predict the baseline night − day difference (genes p < 0.05 in our tissue pseudobulk,
    non-clock), at the inferred times; gene-label permutation p. Night-programme genes (heat shock, IEG) baseline
    difference; interaction genes vs our rhythm sets (Fisher).
Outputs (results/r3_<key>/): bulk_de.csv, bulk_sampling_phase.csv, bulk_validation.csv, bulk_links.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "ma2025"
MOD = f"r3_{KEY}"
W = phase.W
rng = np.random.default_rng(314)
roots = [Path(os.environ["MA_ROOT"])] if KEY == "ma2025dry" else [Path("external/ma2025"), EXTERNAL / "ma2025"]
cands = [p for r in roots if r.exists() for p in r.rglob("*") if p.is_file() and
         ("cra019827" in p.name.lower() or "bulk" in p.name.lower()) and any(s in p.name.lower() for s in (".csv", ".tsv", ".txt"))]
if not cands:
    sys.exit("no bulk count matrix found yet")
f = cands[0]
cnt = pd.read_csv(f, sep=None, engine="python", index_col=0)
names = [f"{c}{m}{i}" for c in "CN" for m in ("MD", "MN") for i in (1, 2, 3)]
cols = {c: next((k for k in cnt.columns if c.lower() in str(k).lower()), None) for c in names}
missing = [c for c, k in cols.items() if k is None]
if missing:
    sys.exit(f"bulk matrix lacks samples {missing}")
cnt = cnt[[cols[c] for c in names]]
cnt.columns = names
if cnt.index.astype(str).str.startswith("ENSMUSG").mean() > 0.5:
    gm = pd.read_csv(EXTERNAL / "resources/ensembl_gene_map.csv").set_index("gene_id")["symbol"]
    cnt.index = cnt.index.astype(str).str.split(".").str[0].map(gm)
    cnt = cnt[cnt.index.notna()].groupby(level=0).sum()
cpm = cnt / cnt.sum(0) * 1e6
L = np.log2(cpm[(cpm > 1).sum(1) >= 3] + 1)
grp = np.array([n[:3] for n in names])
G = ["CMD", "CMN", "NMD", "NMN"]
X = np.column_stack([(grp == g).astype(float) for g in G])
B = np.linalg.lstsq(X, L.to_numpy().T, rcond=None)[0]
res = L.to_numpy().T - X @ B
dfr = len(names) - 4
s2 = (res ** 2).sum(0) / dfr
CON = {"baseline_night_minus_day": {"CMN": 1, "CMD": -1}, "noise_effect_day_group": {"NMD": 1, "CMD": -1},
       "noise_effect_night_group": {"NMN": 1, "CMN": -1}, "noise_x_time_interaction": {"NMN": 1, "CMN": -1, "NMD": -1, "CMD": 1}}
de = []
for k, w in CON.items():
    c = np.array([w.get(g, 0) for g in G], float)
    est = c @ B
    se = np.sqrt(s2 * (c ** 2 / 3).sum())
    t = est / se
    p = 2 * stats.t.sf(np.abs(t), dfr)
    de.append(pd.DataFrame({"contrast": k, "gene": L.index, "log2FC": est, "t": t, "p": p,
                            "q_BH": multipletests(np.nan_to_num(p, nan=1), method="fdr_bh")[1], "mean_log2cpm": L.mean(1).to_numpy()}))
de = pd.concat(de, ignore_index=True)
write_csv(de, MOD, "bulk_de.csv", f"{KEY} bulk: baseline night − day, noise effects by exposure time, interaction (log2 CPM OLS)",
          required=["contrast", "gene", "log2FC", "p", "q_BH"])

off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
O_E = float(off[(off["dataset"] == "GSE88954") & (off["condition"] == "sham") & (off["unit"] == "bulk")]["offset_canonical_h"].iloc[0])
e88 = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv").query("condition == 'sham'").drop_duplicates("gene").set_index("gene")
tp = pd.read_csv(RESULTS / "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").set_index("gene")
clock = [g for g in phase.CANONICAL_CLOCK if g in L.index]
Z = L.loc[clock].sub(L.loc[clock].mean(1), axis=0)
rows = []
for s in names:
    for ref, fit, o in [("GSE88954 sham clock fits", e88, O_E), ("our cochlear tissue-pseudobulk clock fits", tp, O_C)]:
        g = [x for x in clock if x in fit.index]
        best = max(((t, np.corrcoef(Z.loc[g, s], fit.loc[g, "a"] * np.cos(W * (t + o)) + fit.loc[g, "b"] * np.sin(W * (t + o)))[0, 1])
                    for t in np.arange(0, 24, 0.25)), key=lambda x: x[1])
        rows.append({"sample": s, "group": s[:3], "reference": ref, "inferred_internal_h": best[0], "r": best[1]})
sp_ = pd.DataFrame(rows)
for (gname, ref), g in sp_.groupby(["group", "reference"]):
    m, R = phase.circ_mean(g["inferred_internal_h"])
    sp_.loc[g.index, "group_mean_h"] = m
    sp_.loc[g.index, "group_resultant"] = R
write_csv(sp_, MOD, "bulk_sampling_phase.csv", f"{KEY} bulk: inferred sampling phase per sample from core-clock genes",
          required=["sample", "group", "reference", "inferred_internal_h", "r"])

ref_e = sp_[sp_["reference"].str.startswith("GSE88954")].drop_duplicates("group").set_index("group")["group_mean_h"]
t_day, t_night = float(ref_e["CMD"]), float(ref_e["CMN"])
obs = de[de["contrast"] == "baseline_night_minus_day"].set_index("gene")["log2FC"]
val = []
for label, sel in [("our cochlear tissue pseudobulk p<0.05, non-clock", tp[(tp["p"] < 0.05) & ~tp.index.isin(phase.CANONICAL_CLOCK)]),
                   ("GSE88954 sham p<0.05 (comparator)", e88[e88["p"] < 0.05])]:
    o = O_C if label.startswith("our") else O_E
    g = sel.index.intersection(obs.index)
    pdv = (sel.loc[g, "a"] * (np.cos(W * (t_night + o)) - np.cos(W * (t_day + o))) +
           sel.loc[g, "b"] * (np.sin(W * (t_night + o)) - np.sin(W * (t_day + o)))).to_numpy()
    ob = obs.loc[g].to_numpy()
    r = stats.pearsonr(pdv, ob)[0]
    null = np.array([stats.pearsonr(pdv, rng.permutation(ob))[0] for _ in range(2000)])
    val.append({"predictor": label, "day_internal_h": t_day, "night_internal_h": t_night, "n_genes": len(g), "pearson_r": r,
                "p_perm": float((1 + (null >= r).sum()) / 2001)})
write_csv(pd.DataFrame(val), MOD, "bulk_validation.csv",
          f"{KEY} bulk: do our cochlear rhythms predict the baseline night − day difference at the inferred sampling times?",
          required=["predictor", "n_genes", "pearson_r", "p_perm"])
HSG = ["Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"]
IEG = ["Fos", "Fosb", "Jun", "Junb", "Egr1", "Egr2", "Egr3", "Ier2", "Nr4a1", "Dusp1", "Atf3", "Klf2", "Klf4", "Zfp36"]
lk = []
for k in ["baseline_night_minus_day", "noise_x_time_interaction"]:
    d = de[de["contrast"] == k].set_index("gene")
    for name, gs in [("heat shock", HSG), ("IEG", IEG), ("our cochlear tissue rhythmic p<0.05", list(tp.index[tp["p"] < 0.05]))]:
        x = d[d.index.isin(gs)]
        if len(x) < 3:
            continue
        lk.append({"contrast": k, "gene_set": name, "n_genes": len(x), "mean_log2FC": float(x["log2FC"].mean()),
                   "frac_positive": float((x["log2FC"] > 0).mean()),
                   "p_wilcoxon_vs_rest": float(stats.mannwhitneyu(x["log2FC"], d.loc[~d.index.isin(gs), "log2FC"]).pvalue),
                   "n_p01": int((x["p"] < 0.01).sum())})
write_csv(pd.DataFrame(lk), MOD, "bulk_links.csv", f"{KEY} bulk: night-programme genes and our rhythm sets in the baseline and interaction contrasts",
          required=["contrast", "gene_set", "mean_log2FC"])
log("done")
