"""Round 3, WP2 — Ma 2025 (CRA019820): does time-of-day noise sensitivity follow the cochlear rhythms of the atlas?

Usage: python scripts/312_ma2025_rhythm_links.py [ma2025 | ma2025dry]
Design: cochleae 3 d after 2 h of noise at ZT2–4 (day_noise) or ZT14–16 (night_noise), and controls; pooled mice.
(a) Sampling-phase QC: per sample, pseudobulk of all cells → centred log CPM of core-clock genes → phase by projection
    on canonical peak phases (φ = arg Σ z_g e^{iωφ_g}) and by template matching with our cochlear tissue-pseudobulk
    clock fits (time maximising the correlation of the sample's centred clock profile with the fitted profile at t).
    If day- and night-noise samples differ in inferred phase, collection time is confounded with exposure time.
(b) Gene responses (311, night_noise_vs_day_noise per label): Fisher enrichment of nominal DE genes (p < 0.01) among our
    rhythm sets (same-label cochlear Tier 1/2; cochlear tissue pseudobulk p < 0.05; GSE88954 sham rhythmic; GSE107086
    noise × time genes p < 0.01; same-label BM Tier 1/2 for marrow-derived labels); directional test: night − day logFC
    of genes whose clock-aligned peak lies in the active (12–24 h) vs rest phase (Wilcoxon).
(c) The paper's NLRP3 / IL-1 axis (Nlrp3, Il1b, Il18, Casp1, Pycard, Gsdmd, Il1rn, Il1r2) in macrophages / monocytes:
    night − day logFC next to the same genes' rhythms in our cochlear cell types (peak ZT, p).
Outputs (results/r3_<key>/): sampling_phase.csv, rhythm_gene_links.csv, nlrp3_axis.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "ma2025"
MOD = f"r3_{KEY}"
W = phase.W
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
O_B = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "bm") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
O_E = float(off[(off["dataset"] == "GSE88954") & (off["condition"] == "sham") & (off["unit"] == "bulk")]["offset_canonical_h"].iloc[0])

# ---- (a) sampling phase
pb = pd.read_parquet(CACHE / "r3_query" / f"{KEY}_pseudobulk.parquet")
des = pd.read_csv(CACHE / "r3_query" / f"{KEY}_pseudobulk_design.csv")
tot = pb.T.groupby(des.set_index("column").loc[pb.columns, "sample"].to_numpy()).sum().T
cpm = tot / tot.sum(0) * 1e6
lc = np.log2(cpm + 1)
clock = [g for g in phase.CANONICAL_CLOCK if g in lc.index and cpm.loc[g].mean() > 1]
z = lc.loc[clock].sub(lc.loc[clock].mean(1), axis=0)
z = z.div(z.std(1).replace(0, np.nan), axis=0).fillna(0)
E = np.exp(1j * W * np.array([phase.CANONICAL_CLOCK[g] for g in clock]))
tp = pd.read_csv(RESULTS / "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").set_index("gene")
tg = [g for g in clock if g in tp.index]
cond = des.drop_duplicates("sample").set_index("sample")["condition"]
rows = []
grid = np.arange(0, 24, 0.25)
for s in lc.columns:
    v = (z[s].to_numpy() * E).sum()
    prof = z.loc[tg, s].to_numpy()
    best_t, best_r = np.nan, -2
    for t in grid:
        pred = tp.loc[tg, "a"] * np.cos(W * (t + O_C)) + tp.loc[tg, "b"] * np.sin(W * (t + O_C))
        r = np.corrcoef(prof, pred - pred.mean())[0, 1] if prof.std() > 0 else np.nan
        if r > best_r:
            best_t, best_r = t, r
    rows.append({"sample": s, "condition": cond.get(s, ""), "n_clock_genes": len(clock),
                 "phase_canonical_projection_h": float(np.mod(np.angle(v) / W, 24)), "projection_length": float(np.abs(v)),
                 "phase_template_match_h": float(best_t), "template_match_r": float(best_r)})
sp_ = pd.DataFrame(rows)
for c, g in sp_.groupby("condition"):
    m, R = phase.circ_mean(g["phase_canonical_projection_h"])
    sp_.loc[g.index, "condition_mean_phase_h"] = m
    sp_.loc[g.index, "condition_resultant"] = R
write_csv(sp_, MOD, "sampling_phase.csv",
          f"{KEY}: inferred sampling phase per sample from core-clock genes (canonical projection; template matching with our "
          "cochlear tissue fits)", required=["sample", "condition", "phase_canonical_projection_h", "phase_template_match_h"])
log("\n" + sp_.round(2).to_string())

# ---- (b) rhythm sets vs night − day responses
de_p = RESULTS / MOD / "pseudobulk_de.csv"
rows = []
if de_p.exists():
    de = pd.read_csv(de_p)
    de = de[de["contrast"] == "night_noise_vs_day_noise"]
    t33c = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_cochlea.csv", usecols=["group", "gene", "tier", "peak_zt", "p"])
    t33b = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier"])
    e88 = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv", usecols=["condition", "gene", "p"])
    g107 = pd.read_csv(RESULTS / "r2_pilots/pilot_GSE107086_de.csv")
    sets_global = {"cochlear tissue pseudobulk p<0.05": set(tp.index[tp["p"] < 0.05]),
                   "GSE88954 sham rhythmic p<0.05": set(e88.loc[(e88["condition"] == "sham") & (e88["p"] < 0.05), "gene"]),
                   "GSE107086 noise x time p<0.01": set(g107.loc[(g107["contrast"] == "noise_night_vs_day_shamsurg") & (g107["p"] < 0.01), "gene"])}
    for lab, g in de.groupby("label"):
        bg = set(g["gene"])
        hit = set(g.loc[g["p"] < 0.01, "gene"])
        sets = dict(sets_global)
        sets["same-label cochlear Tier 1/2"] = set(t33c.loc[(t33c["group"] == lab) & t33c["tier"].isin(["tier1", "tier2"]), "gene"])
        sets["same-label BM Tier 1/2"] = set(t33b.loc[(t33b["group"] == lab) & t33b["tier"].isin(["tier1", "tier2"]), "gene"])
        for name, s_ in sets.items():
            s_ = s_ & bg
            if len(s_) < 5:
                continue
            a, b, c, d = len(hit & s_), len(hit - s_), len(s_ - hit), len(bg - hit - s_)
            orr, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append({"label": lab, "test": "enrichment of night-vs-day DE genes (p<0.01)", "gene_set": name,
                         "n_set_tested": len(s_), "n_hits": len(hit), "overlap": a, "statistic": orr, "p": p})
        r = t33c[(t33c["group"] == lab) & (t33c["p"] < 0.05)].copy()
        if len(r) >= 20:
            r["internal"] = phase.wrap24(r["peak_zt"] - O_C)
            x = g.set_index("gene")["logFC"]
            act = x.reindex(r.loc[r["internal"] >= 12, "gene"]).dropna()
            rest = x.reindex(r.loc[r["internal"] < 12, "gene"]).dropna()
            if len(act) >= 5 and len(rest) >= 5:
                rows.append({"label": lab, "test": "night − day logFC: active-phase-peaking minus rest-phase-peaking genes",
                             "gene_set": "same-label cochlear rhythmic p<0.05", "n_set_tested": len(act) + len(rest),
                             "n_hits": np.nan, "overlap": np.nan, "statistic": float(act.median() - rest.median()),
                             "p": float(stats.mannwhitneyu(act, rest).pvalue)})
rl = pd.DataFrame(rows)
if len(rl):
    rl["q_BH"] = stats.false_discovery_control(rl["p"].to_numpy())
    write_csv(rl, MOD, "rhythm_gene_links.csv", f"{KEY}: night- vs day-noise responses vs our rhythm sets (enrichment, direction)",
              required=["label", "test", "gene_set", "statistic", "p"])

# ---- (c) NLRP3 / IL-1 axis
AX = ["Nlrp3", "Il1b", "Il18", "Casp1", "Pycard", "Gsdmd", "Il1rn", "Il1r2"]
ax = []
if de_p.exists():
    de_all = pd.read_csv(de_p)
    t33c = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_cochlea.csv", usecols=["group", "gene", "peak_zt", "p", "ptt_fc"])
    for lab in ["Macrophage", "Monocyte (Classical)", "Monocyte (Non-classical)", "Neutrophil (Activated)", "Neutrophil (Aged)"]:
        for gname in AX:
            x = de_all[(de_all["label"] == lab) & (de_all["gene"] == gname)]
            rr = t33c[(t33c["group"] == lab) & (t33c["gene"] == gname)]
            row = {"label": lab, "gene": gname}
            for r in x.itertuples():
                row[f"logFC_{r.contrast}"] = r.logFC
                row[f"p_{r.contrast}"] = r.p
            if len(rr):
                row.update({"our_peak_zt": float(rr["peak_zt"].iloc[0]), "our_peak_clock": float(phase.zt_to_clock(rr["peak_zt"].iloc[0])),
                            "our_p": float(rr["p"].iloc[0]), "our_ptt_fc": float(rr["ptt_fc"].iloc[0])})
            if len(row) > 2:
                ax.append(row)
if ax:
    write_csv(pd.DataFrame(ax), MOD, "nlrp3_axis.csv", f"{KEY}: NLRP3 / IL-1 axis — noise responses by exposure time and our cochlear rhythms",
              required=["label", "gene"])
log("done")
