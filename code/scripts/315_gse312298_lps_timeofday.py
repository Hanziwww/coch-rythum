"""Round 3, WP2 — LPS at two times of day in sorted bone-marrow and blood neutrophils (GSE312298).

Design (GEO): neutrophils sorted from BM and peripheral blood 4 h after LPS or PBS given at ZT10 ("4", afternoon) or
ZT18 ("12", midnight); 4 experiments (Ex_143/144 = ZT10, Ex_147/149 = ZT18, verified against the authors' DESeq2
contrast, r = 0.995); suffix 1–2 = PBS, 3–4 = LPS. Time of day is confounded with experiment (2 experiments per time).
Collection = injection + 4 h, i.e. ZT14 vs ZT22 by design.
(a) Per tissue: log2 CPM (prior 1), per-gene OLS with group means → night − day (PBS), LPS effect at each time, and the
    LPS × time interaction (t tests, BH).
(b) Collection-time check: the night − day difference of clock genes predicted for day time t (night = t + 8) from
    (i) canonical clock phases (unit amplitude) and (ii) our BM neutrophil-state cosinor fits, correlated with the
    observed PBS night − day difference; scanned over t.
(c) Validation of our BM neutrophil rhythms (selection-free Tier 1 of the neutrophil states and GMP; per gene mean of
    the state fits): predicted vs observed PBS night − day in BM neutrophils at the design times (ZT14 vs ZT22) and at
    the clock-inferred time; gene-label permutation p; GSE198770 bulk-BM fits as a comparator predictor.
(d) Time-of-day dependence of the LPS response vs our rhythms: Fisher enrichment of interaction genes (p < 0.01) among
    our BM neutrophil Tier-1/2 genes; correlation of the interaction with our predicted baseline difference.
Outputs (results/r3_response/): gse312298_de.csv, gse312298_collection_time.csv, gse312298_validation.csv,
gse312298_lps_timeofday_links.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

warnings.filterwarnings("ignore")
MOD = "r3_response"
W = phase.W
rng = np.random.default_rng(315)
d = pd.read_excel(EXTERNAL / "geo/GSE312298/GSE312298_DESeq2_All_togother_umi_counts_20220307_105325.xlsx")
raw = d.set_index(d.columns[0])[[c for c in d.columns if c.endswith(".raw")]]
raw = raw[~raw.index.duplicated()]
meta = pd.DataFrame({"col": raw.columns})
p_ = meta["col"].str.extract(r"Ex_(\d+)_(BM|PB)_(\d)\.raw")
meta["experiment"], meta["tissue"], meta["k"] = p_[0].astype(int), p_[1], p_[2].astype(int)
meta["time"] = np.where(meta["experiment"].isin([143, 144]), "day", "night")        # injection ZT10 vs ZT18
meta["treatment"] = np.where(meta["k"] <= 2, "PBS", "LPS")
meta["group"] = meta["treatment"] + "_" + meta["time"]
DESIGN_DAY_ZT, DESIGN_NIGHT_ZT = 14.0, 22.0                                         # collection = injection + 4 h
CONTRASTS = {"night_minus_day_PBS": [("PBS_night", 1), ("PBS_day", -1)],
             "LPS_effect_day": [("LPS_day", 1), ("PBS_day", -1)],
             "LPS_effect_night": [("LPS_night", 1), ("PBS_night", -1)],
             "LPS_x_night_interaction": [("LPS_night", 1), ("PBS_night", -1), ("LPS_day", -1), ("PBS_day", 1)]}

de, logc = [], {}
for t, mt in meta.groupby("tissue"):
    c = raw[mt["col"]]
    cpm = c / c.sum(0) * 1e6
    keep = (cpm > 5).sum(1) >= 4
    L = np.log2(cpm[keep] + 1)
    logc[t] = L
    groups = mt["group"].to_numpy()
    G = sorted(set(groups))
    X = np.column_stack([(groups == g).astype(float) for g in G])
    B = np.linalg.lstsq(X, L.to_numpy().T, rcond=None)[0]
    res = L.to_numpy().T - X @ B
    dfr = X.shape[0] - X.shape[1]
    s2 = (res ** 2).sum(0) / dfr
    XtXi = np.linalg.inv(X.T @ X)
    for name, terms in CONTRASTS.items():
        cvec = np.zeros(len(G))
        for g, w in terms:
            cvec[G.index(g)] = w
        est = cvec @ B
        se = np.sqrt(s2 * (cvec @ XtXi @ cvec))
        tt = est / np.where(se > 0, se, np.nan)
        pv = 2 * stats.t.sf(np.abs(tt), dfr)
        de.append(pd.DataFrame({"tissue": t, "contrast": name, "gene": L.index, "log2FC": est, "t": tt, "p": pv,
                                "q_BH": multipletests(np.nan_to_num(pv, nan=1.0), method="fdr_bh")[1],
                                "mean_log2cpm": L.mean(1).to_numpy()}))
de = pd.concat(de, ignore_index=True)
write_csv(de, MOD, "gse312298_de.csv", "GSE312298 sorted BM / blood neutrophils: night − day (PBS), LPS effects by time, "
          "LPS × time interaction (log2 CPM OLS, t tests)", required=["tissue", "contrast", "gene", "log2FC", "p", "q_BH"])
log("\n" + de.groupby(["tissue", "contrast"]).apply(lambda x: pd.Series({"q05": int((x["q_BH"] < 0.05).sum()), "p01": int((x["p"] < 0.01).sum())})).to_string())

# ---- our BM neutrophil fits (selection-free Tier 1 of neutrophil states + GMP)
NEU = ["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Band)", "Neutrophil (Mature)", "Granulocyte-Monocyte Progenitor"]
t33 = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier", "a", "b", "p"])
neu_all = t33[t33["group"].isin(NEU)]
neu_fit = neu_all.groupby("gene")[["a", "b"]].mean()
neu_t1 = neu_all[neu_all["tier"] == "tier1"].groupby("gene")[["a", "b"]].mean()
neu_t12 = set(neu_all.loc[neu_all["tier"].isin(["tier1", "tier2"]), "gene"])
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
D_BM = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "bm") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
D_198 = float(off[(off["dataset"] == "GSE198770_bone_marrow") & (off["condition"] == "control") & (off["unit"] == "bulk")]["offset_canonical_h"].iloc[0])
e198 = pd.read_csv(RESULTS / "r2_external/external_rhythm_GSE198770_bone_marrow.csv", usecols=["condition", "gene", "a", "b", "p"])
e198 = e198[e198["condition"] == "control"].drop_duplicates("gene").set_index("gene")


def pred_diff(fit, t_day_internal, offset, gap=8.0):
    """night − day difference predicted at internal day time t (night = t + gap) from cosinor (a, b) on its own axis."""
    td, tn = t_day_internal + offset, t_day_internal + gap + offset
    return fit["a"] * (np.cos(W * tn) - np.cos(W * td)) + fit["b"] * (np.sin(W * tn) - np.sin(W * td))


obs_bm = de[(de["tissue"] == "BM") & (de["contrast"] == "night_minus_day_PBS")].set_index("gene")["log2FC"]
clock = [g for g in phase.CANONICAL_CLOCK if g in obs_bm.index]
canon = pd.DataFrame({"a": [np.cos(W * phase.CANONICAL_CLOCK[g]) for g in clock],
                      "b": [np.sin(W * phase.CANONICAL_CLOCK[g]) for g in clock]}, index=clock)
ct_rows = []
for ref, fit, o in [("canonical clock phases", canon, 0.0), ("our BM neutrophil-state fits (clock genes)", neu_fit.reindex(clock).dropna(), D_BM)]:
    for tday in np.arange(0, 24, 0.5):
        pdv = pred_diff(fit, tday, o)
        ct_rows.append({"reference": ref, "n_clock_genes": len(fit), "day_collection_internal_h": tday,
                        "pearson_r": stats.pearsonr(pdv, obs_bm.loc[fit.index])[0]})
ct = pd.DataFrame(ct_rows)
best = ct.loc[ct.groupby("reference")["pearson_r"].idxmax()]
ct["is_best"] = ct.index.isin(best.index)
write_csv(ct, MOD, "gse312298_collection_time.csv",
          "GSE312298 BM neutrophils (PBS): clock-gene night − day difference vs prediction by assumed day collection time "
          "(night = day + 8 h); design ZT14 vs ZT22", required=["reference", "day_collection_internal_h", "pearson_r"])
t_clock = float(best[best["reference"] == "canonical clock phases"]["day_collection_internal_h"].iloc[0])
log("collection-time scan best: " + "; ".join(f"{r.reference} {r.day_collection_internal_h} (r {r.pearson_r:.2f})" for r in best.itertuples()))

val = []
for label, tday in [("design (ZT14 vs ZT22)", DESIGN_DAY_ZT), ("clock-inferred", t_clock)]:
    for name, fit, o in [("our BM neutrophil-state Tier 1 (selection-free)", neu_t1, D_BM),
                         ("our BM neutrophil-state fits, all tested genes", neu_fit, D_BM),
                         ("GSE198770 bulk BM control, p < 0.05 (comparator)", e198[e198["p"] < 0.05], D_198)]:
        g = fit.index.intersection(obs_bm.index)
        pdv = pred_diff(fit.loc[g], tday, o).to_numpy()
        ob = obs_bm.loc[g].to_numpy()
        r = stats.pearsonr(pdv, ob)[0]
        null = np.array([stats.pearsonr(pdv, rng.permutation(ob))[0] for _ in range(2000)])
        val.append({"collection_time": label, "day_internal_h": tday, "predictor": name, "n_genes": len(g), "pearson_r": r,
                    "sign_agreement": float((np.sign(pdv) == np.sign(ob)).mean()), "p_perm": float((1 + (null >= r).sum()) / 2001)})
val = pd.DataFrame(val)
write_csv(val, MOD, "gse312298_validation.csv",
          "Independent check of our BM neutrophil rhythms: predicted vs observed night − day difference in sorted BM neutrophils (PBS)",
          required=["collection_time", "predictor", "n_genes", "pearson_r", "p_perm"])
log("\n" + val.round(3).to_string())

links = []
for t in ["BM", "PB"]:
    x = de[(de["tissue"] == t) & (de["contrast"] == "LPS_x_night_interaction")].set_index("gene")
    bg = set(x.index)
    hit = set(x.index[x["p"] < 0.01])
    s_ = neu_t12 & bg
    a, b, c, dd = len(hit & s_), len(hit - s_), len(s_ - hit), len(bg - hit - s_)
    orr, p = stats.fisher_exact([[a, b], [c, dd]], alternative="greater")
    g = neu_fit.index.intersection(x.index)
    rho, pr = stats.spearmanr(pred_diff(neu_fit.loc[g], DESIGN_DAY_ZT, D_BM), x.loc[g, "log2FC"])
    links.append({"tissue": t, "n_interaction_p01": len(hit), "n_our_neutrophil_tier12_expressed": len(s_), "overlap": a,
                  "odds_ratio": orr, "p_fisher_greater": p, "spearman_interaction_vs_predicted_baseline_diff": rho,
                  "p_spearman": pr})
write_csv(pd.DataFrame(links), MOD, "gse312298_lps_timeofday_links.csv",
          "Time-of-day dependence of the neutrophil LPS response vs our BM neutrophil rhythms",
          required=["tissue", "overlap", "odds_ratio", "p_fisher_greater"])
log("done")
