"""Round 3, WP4 — is the night immediate-early / NF-κB component a handling effect, and which night rhythms are
glucocorticoid-dependent?

(a) Sample-level handling component: the per-sample × cell-type dissociation-stress score (01_technical_audit, IEG set of
    12_technical_audit.py) decomposed by OLS into cell-type + sample effects per tissue; share of variance carried by the
    sample term, cosinor of the sample effects (ZT), correlation with the tissue-wide sample pattern (210, PC1).
(b) Gene-level flags: for every Tier-1/2 call (33) and cochlear Tier C call (211), correlation of the gene's centred
    log2-CPM profile with the same cell type's stress profile over the 6 samples; 'stress-like' if r > 0.8. Counts by
    tissue, tier, gene class (IEG / heat shock / NF-κB / other) and phase bin; tiers recounted without stress-like calls.
(c) Glucocorticoid dependence: external cochlea GSE88954 — 'GC-dependent rhythmic' = sham p < 0.05 and ADX p > 0.2,
    'GC-independent' = p < 0.05 in both. Among our rhythmic genes (BM Tier 1/2; cochlear tissue pseudobulk p < 0.05),
    Fisher tests of GC-dependence for night (internal phase 12–24 h) vs day peaks; phases of the curated
    glucocorticoid-response set and of Per1 / Nfil3 per tissue vs canonical.
Outputs (results/r3_audit/): handling_sample_effect.csv, handling_gene_flags.csv, gc_dependence_by_phase.csv,
gc_response_phases.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r3_audit"
IEG = {"Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36", "Dusp1",
       "Klf2", "Klf4", "Klf6", "Btg2", "Ppp1r15a", "Gadd45b", "Socs3", "Cebpd"}
HSG = {"Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"}
NFKB = {"Nfkbia", "Nfkbiz", "Tnfaip3", "Cxcl2", "Ccl3", "Ccl4", "Nfkbid", "Rel", "Relb", "Il1b"}
tech = pd.read_csv(RESULTS / "01_technical_audit/technical_audit.csv")
stress = tech[tech["metric"] == "stress_log2cpm"].copy()
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
OFF = {t: float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == t) & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
       for t in ["bm", "cochlea"]}

# ---- (a) sample-level handling component
rows = []
for t, g in stress.groupby("tissue"):
    g = g[g["status"] == "primary"]
    m = smf.ols("value ~ C(cell_type) + C(sample_id)", g).fit()
    an = sm.stats.anova_lm(m, typ=2)
    eff = g.assign(resid_ct=g["value"] - g.groupby("cell_type")["value"].transform("mean")).groupby(["sample_id", "zt"])["resid_ct"].mean().reset_index().sort_values("zt")
    f = cosinor.fit(eff[["resid_ct"]].to_numpy(), eff["zt"].to_numpy(float))
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"]
    r = float(np.corrcoef(eff["resid_ct"], eff["sample_id"].map(pc1))[0, 1])
    share = float(an.loc["C(sample_id)", "sum_sq"] / an["sum_sq"].sum())
    for s in eff.itertuples():
        rows.append({"tissue": t, "sample_id": s.sample_id, "zt": s.zt, "clock": float(phase.zt_to_clock(s.zt)),
                     "sample_effect_stress": s.resid_ct, "sample_term_variance_share": share,
                     "p_sample_term": float(an.loc["C(sample_id)", "PR(>F)"]),
                     "sample_effect_peak_zt": float(f["peak"][0]), "sample_effect_cosinor_p": float(f["p"][0]),
                     "r_with_tissue_pattern_pc1": r})
se = pd.DataFrame(rows)
write_csv(se, MOD, "handling_sample_effect.csv",
          "Dissociation-stress score: sample effect (cell-type-centred), variance share, ZT cosinor, link to the sample pattern",
          required=["tissue", "sample_id", "zt", "sample_effect_stress", "sample_term_variance_share"])
log("\n" + se.drop_duplicates("tissue")[["tissue", "sample_term_variance_share", "p_sample_term", "sample_effect_peak_zt",
                                          "sample_effect_cosinor_p", "r_with_tissue_pattern_pc1"]].round(3).to_string())

# ---- (b) gene-level stress-like flags
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
calls = []
for t in ["bm", "cochlea"]:
    tb = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "tier", "peak_zt"])
    tb = tb[tb["tier"].isin(["tier1", "tier2"])].assign(tissue=t)
    calls.append(tb)
tcp = RESULTS / "r2_cochlea/cochlea_relaxed_tier.csv"
if tcp.exists():
    tc = pd.read_csv(tcp, usecols=["cell_type", "gene", "tierC", "peak_zt"])
    tc = tc[tc["tierC"].astype(str) == "True"].rename(columns={"cell_type": "group"}).drop(columns="tierC")
    calls.append(tc.assign(tier="tierC", tissue="cochlea"))
calls = pd.concat(calls, ignore_index=True)
flag_rows = []
for t in ["bm", "cochlea"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    samples = d0.drop_duplicates("sample_id").sort_values("zt")["sample_id"].tolist()
    st = stress[stress["tissue"] == t].pivot_table(index="cell_type", columns="sample_id", values="value")
    for ct, g in calls[calls["tissue"] == t].groupby("group"):
        cols = [f"{ct}|{s}" for s in samples]
        if not set(cols) <= set(pb.columns) or ct not in st.index:
            continue
        sub = pb.loc[pb.index.intersection(g["gene"]), cols].to_numpy(float)
        lib = pb[cols].to_numpy(float).sum(0)
        l = np.log2(sub / lib * 1e6 + 1)
        l = l - l.mean(1, keepdims=True)
        s = st.loc[ct, samples].to_numpy(float)
        s = s - s.mean()
        den = np.sqrt((l ** 2).sum(1) * (s ** 2).sum())
        r = np.where(den > 0, (l * s).sum(1) / np.where(den > 0, den, 1), np.nan)
        rmap = dict(zip(pb.index.intersection(g["gene"]), r))
        for x in g.itertuples():
            flag_rows.append({"tissue": t, "cell_type": ct, "gene": x.gene, "tier": x.tier, "peak_zt": x.peak_zt,
                              "r_with_stress_profile": rmap.get(x.gene, np.nan)})
fl = pd.DataFrame(flag_rows)
fl["stress_like"] = fl["r_with_stress_profile"] > 0.8
fl["gene_class"] = np.select([fl["gene"].isin(IEG), fl["gene"].isin(HSG), fl["gene"].isin(NFKB)], ["IEG", "heat_shock", "NF-kB"], "other")
fl["phase_bin"] = pd.cut(fl["peak_zt"] % 24, [0, 6, 12, 18, 24], right=False, labels=["ZT0-6", "ZT6-12", "ZT12-18", "ZT18-24"])
summ = fl.groupby(["tissue", "tier", "gene_class", "phase_bin"], observed=True).agg(
    n_calls=("gene", "size"), n_stress_like=("stress_like", "sum"), median_r=("r_with_stress_profile", "median")).reset_index()
summ["frac_stress_like"] = summ["n_stress_like"] / summ["n_calls"]
tot = fl.groupby(["tissue", "tier"]).agg(n_calls=("gene", "size"), n_stress_like=("stress_like", "sum")).reset_index()
tot["gene_class"], tot["phase_bin"] = "ALL", "ALL"
tot["frac_stress_like"] = tot["n_stress_like"] / tot["n_calls"]
tot["n_calls_without_stress_like"] = tot["n_calls"] - tot["n_stress_like"]
write_csv(pd.concat([tot, summ], ignore_index=True), MOD, "handling_gene_flags.csv",
          "Tier-1/2 and Tier C calls whose profile follows the cell type's dissociation-stress profile (r > 0.8), by class and phase",
          required=["tissue", "tier", "gene_class", "phase_bin", "n_calls", "n_stress_like"])
log("\n" + tot.round(3).to_string())

# ---- (c) glucocorticoid dependence (external cochlea sham vs ADX)
e = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv", usecols=["condition", "gene", "p"]).drop_duplicates(["condition", "gene"])
ew = e.pivot(index="gene", columns="condition", values="p")
gc_dep = set(ew.index[(ew["sham"] < 0.05) & (ew["adx"] > 0.2)])
gc_ind = set(ew.index[(ew["sham"] < 0.05) & (ew["adx"] < 0.05)])
ours = {"bm": pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier", "peak_zt"])
        .query("tier in ['tier1', 'tier2']"),
        "cochlea": pd.read_csv(RESULTS / "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").query("p < 0.05")}
gd = []
for t, d in ours.items():
    d = d.drop_duplicates("gene").copy()
    d["internal"] = phase.wrap24(d["peak_zt"] - OFF[t])
    d = d[d["gene"].isin(gc_dep | gc_ind)]
    night = d["internal"] >= 12
    dep = d["gene"].isin(gc_dep)
    a, b, c, dd = int((night & dep).sum()), int((night & ~dep).sum()), int((~night & dep).sum()), int((~night & ~dep).sum())
    orr, p = stats.fisher_exact([[a, b], [c, dd]])
    gd.append({"tissue": t, "gene_set": "BM Tier 1/2 (genes)" if t == "bm" else "cochlear tissue pseudobulk p<0.05",
               "n_externally_rhythmic": len(d), "night_gc_dependent": a, "night_gc_independent": b, "day_gc_dependent": c,
               "day_gc_independent": dd, "odds_ratio_night_gc_dependent": orr, "p_fisher": p})
write_csv(pd.DataFrame(gd), MOD, "gc_dependence_by_phase.csv",
          "Are our night-peaking rhythmic genes enriched for externally GC-dependent rhythms (GSE88954 sham rhythmic, lost after ADX)?",
          required=["tissue", "odds_ratio_night_gc_dependent", "p_fisher"])
gcs = set(yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))["systemic"]["glucocorticoid_response"]) \
    if "systemic" in yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml")) else set()
if not gcs:
    y = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
    gcs = set(next(v["glucocorticoid_response"] for v in y.values() if isinstance(v, dict) and "glucocorticoid_response" in v))
ph = []
for t in ["bm", "cochlea"]:
    tb = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "p", "peak_zt"])
    for name, genes in [("glucocorticoid_response (curated)", gcs), ("Per1", {"Per1"}), ("Nfil3", {"Nfil3"}),
                        ("GSE88954 GC-dependent rhythmic", gc_dep), ("GSE88954 GC-independent rhythmic", gc_ind)]:
        x = tb[tb["gene"].isin(genes) & (tb["p"] < 0.05)]
        if len(x) < 3:
            continue
        m, R = phase.circ_mean(x["peak_zt"])
        ph.append({"tissue": t, "gene_set": name, "n_rows_p05": len(x), "n_genes": x["gene"].nunique(), "circ_mean_peak_zt": m,
                   "circ_mean_peak_clock": float(phase.zt_to_clock(m)), "resultant": R,
                   "rayleigh_p": phase.rayleigh(x["peak_zt"].to_numpy())[1],
                   "canonical_peak_zt": phase.CANONICAL_CLOCK.get(name, np.nan)})
write_csv(pd.DataFrame(ph), MOD, "gc_response_phases.csv",
          "Peak phases of glucocorticoid-response genes, Per1, Nfil3 and externally GC-dependent / independent rhythmic genes",
          required=["tissue", "gene_set", "circ_mean_peak_zt", "resultant"])
log("done")
