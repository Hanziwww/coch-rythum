"""Round 2, WP6 pilot — linking public cochlear noise-response data (bulk and GeoMx spatial) to the atlas.

(A) Collection time of GSE107086 (sham noise, sham surgery, night − day): GEO lists ZT11/ZT23, the series design
    implies ~ZT6/ZT18 (noise at ZT3/ZT15 + 2 h). The night − day difference of a cosinor gene at day time t is
    Δ(t) = −2 (a cos ωt + b sin ωt); the day time maximising the correlation of predicted and observed Δ over
    reference-rhythmic genes is estimated with GSE88954 sham (same laboratory and tissue; clock genes and all
    rhythmic genes) — independent of our atlas.
(B) Validation of our cochlear rhythms: at that time, predicted Δ from our tissue-level pseudobulk cosinor (251) vs the
    observed night − day difference without noise, over genes rhythmic in our cochlea (p < 0.05), all and non-clock;
    gene-label permutation null (2000).
(C) Time-of-day dependence of the noise response: overlap of noise × time interaction genes (p < 0.01) with our gene
    sets (Fisher, expressed background); correlation of the interaction with the baseline night − day difference
    predicted by our rhythms; immune/structural signature scores (our cochlear markers) per sample with OLS
    score ~ noise × time × surgery.
(D) Noise-response kinetics (GSE312476, 0/4/8 h after 2 h of 100 dB): mean log2FC of IEG, heat-shock, NF-κB, IL-1
    sets and of our night- vs day-peaking cochlear genes.
(E) GeoMx (GSE312646; spiral-ganglion-neuron and supporting-cell AOIs, noise vs control): our cell-type signature
    scores per AOI (localisation = positive control; noise effect per region) and gene-set noise effects per region;
    localisation of our Tier C genes (calibrated) of neuronal vs epithelial types.
(B2) Independent check of the cochlear Tier C (211; selected without GSE107086): per gene, the mean night − day
    difference predicted by its cell-type fits where it is called, vs the observed difference; contrasts: R0 calls
    outside Tier C and genes tested but not called.
(B3) Night activation programme in an independent cohort: night − day difference without noise of IEG, heat-shock,
    NF-κB and IL-1 genes (sham surgery and ADX), against all expressed genes.
Outputs (results/r2_pilots/): pilot_collection_time_inference.csv, pilot_daynight_rhythm_validation.csv,
pilot_tierC_independent_validation.csv, pilot_night_programme_check.csv,
pilot_noise_geneset_overlap.csv, pilot_noise_signatures.csv, pilot_noise_kinetics_genesets.csv,
pilot_geomx_signatures.csv, pilot_geomx_genesets.csv
"""
import gzip
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import ROOT, RESULTS, log, write_csv  # noqa: E402

MOD = "r2_pilots"
W = phase.W
rng = np.random.default_rng(12)
GEO = ROOT / "external/geo"
IEG = ["Fos", "Fosb", "Jun", "Junb", "Egr1", "Egr2", "Egr3", "Ier2", "Nr4a1", "Dusp1", "Atf3", "Klf2", "Klf4", "Zfp36"]
HSG = ["Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"]
NFKB = ["Nfkbia", "Nfkbiz", "Tnfaip3", "Ier3", "Cxcl2", "Ccl3", "Ccl4", "Nfkbid", "Rel", "Relb"]
IL1 = ["Il1b", "Il1rn", "Il1r2", "Il1a"]
CLOCKG = set(phase.CANONICAL_CLOCK) | {"Bhlhe40", "Bhlhe41"}
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
O_E = float(off[(off["dataset"] == "GSE88954") & (off["condition"] == "sham") & (off["unit"] == "bulk")]["offset_canonical_h"].iloc[0])
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])

de1 = pd.read_csv(RESULTS / "r2_pilots/pilot_GSE107086_de.csv")
de2 = pd.read_csv(RESULTS / "r2_pilots/pilot_GSE312476_de.csv")
ours = pd.read_csv(RESULTS / "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").set_index("gene")
e88 = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv")
e88 = e88[e88["condition"] == "sham"].drop_duplicates("gene").set_index("gene")
base = de1[de1["contrast"] == "night_minus_day_no_noise_shamsurg"].set_index("gene")


def pred_delta(tab, t_internal, offset):
    """night − day difference predicted from cosinor (a, b on the table's own time axis) for day time t (internal)."""
    t = t_internal + offset
    return -2 * (tab["a"] * np.cos(W * t) + tab["b"] * np.sin(W * t))


# ---- (A) collection-time inference with the external cochlea
rows = []
for ref_name, ref, o_ in [("GSE88954 sham (external)", e88, O_E), ("this atlas cochlea pseudobulk", ours, O_C)]:
    for subset in ["clock genes", "rhythmic genes p<0.05"]:
        g = ref[ref["p"] < 0.05] if subset.startswith("rhythmic") else ref[ref.index.isin(CLOCKG)]
        g = g[g.index.isin(base.index)]
        for t in np.arange(0, 24, 0.5):
            pdlt = pred_delta(g, t, o_)
            r = stats.pearsonr(pdlt, base.loc[g.index, "logFC"])[0] if len(g) >= 5 else np.nan
            rows.append({"reference": ref_name, "genes": subset, "n_genes": len(g), "day_collection_internal_h": t,
                         "pearson_r": r})
ct = pd.DataFrame(rows)
best = ct.loc[ct.groupby(["reference", "genes"])["pearson_r"].idxmax()]
ct["is_best"] = ct.index.isin(best.index)
write_csv(ct, MOD, "pilot_collection_time_inference.csv",
          "GSE107086: correlation of predicted vs observed night − day differences (no noise) by assumed day collection "
          "time (internal clock-aligned h; night = day + 12)", required=["reference", "genes", "day_collection_internal_h", "pearson_r"])
# clock genes are the phase markers: the external cochlea's clock genes fix the collection time (independent of our atlas)
t_hat = float(best[(best["reference"].str.startswith("GSE88954")) & (best["genes"] == "clock genes")]["day_collection_internal_h"].iloc[0])
log("best day collection time (internal h): " + "; ".join(f"{r.reference}/{r.genes}: {r.day_collection_internal_h} (r {r.pearson_r:.2f})"
                                                          for r in best.itertuples()))

# ---- (B) validation of our cochlear rhythms at the inferred (and the two documented) collection times
val = []
for label, t in [("inferred from GSE88954", t_hat), ("GEO characteristics ZT11/ZT23", 11.0), ("design ZT6/ZT18", 6.0)]:
    for subset in ["all rhythmic (p<0.05)", "non-clock rhythmic (p<0.05)", "rhythmic p<0.01"]:
        g = ours[ours["p"] < (0.01 if "0.01" in subset else 0.05)]
        if subset.startswith("non-clock"):
            g = g[~g.index.isin(CLOCKG)]
        g = g[g.index.isin(base.index)]
        pdlt = pred_delta(g, t, O_C)
        obs = base.loc[g.index, "logFC"].to_numpy()
        r = stats.pearsonr(pdlt, obs)[0]
        null = np.array([stats.pearsonr(pdlt, rng.permutation(obs))[0] for _ in range(2000)])
        val.append({"collection_time": label, "day_internal_h": t, "genes": subset, "n_genes": len(g), "pearson_r": r,
                    "sign_agreement": float((np.sign(pdlt) == np.sign(obs)).mean()), "p_perm": float((1 + (null >= r).sum()) / 2001)})
val = pd.DataFrame(val)
write_csv(val, MOD, "pilot_daynight_rhythm_validation.csv",
          "Our cochlear tissue-level rhythms predict the independent night − day difference (GSE107086, no noise, sham surgery)",
          required=["collection_time", "genes", "n_genes", "pearson_r", "p_perm"])
log("\n" + val.round(3).to_string())

# ---- (B2) Tier C vs the independent cohort
tc_path = RESULTS / "r2_cochlea/cochlea_relaxed_tier.csv"
if tc_path.exists():
    tcf = pd.read_csv(tc_path, usecols=["cell_type", "gene", "a", "b", "p", "tierC", "tierC_prespecified", "s_external", "s_bm"])
    for c in ["tierC", "tierC_prespecified", "s_external", "s_bm"]:
        tcf[c] = tcf[c].astype(str) == "True"
    e_pred = pred_delta(e88, t_hat, O_E)                 # GSE88954's own prediction (the external information R1 can import)
    tv = []
    for label, mask in [("Tier C (calibrated rule)", tcf["tierC"]),
                        ("Tier C, BM-phase support only (no external-cochlea information)", tcf["tierC"] & tcf["s_bm"] & ~tcf["s_external"]),
                        ("Tier C, external-cochlea support", tcf["tierC"] & tcf["s_external"]),
                        ("R0 calls outside Tier C (edgeR support only)", tcf["tierC_prespecified"] & ~tcf["tierC"]),
                        ("tested, p >= 0.05", tcf["p"] >= 0.05)]:
        g = tcf[mask]
        per_gene = pd.Series(pred_delta(g, t_hat, O_C).to_numpy(), index=g["gene"].to_numpy()).groupby(level=0).mean()
        per_gene = per_gene[per_gene.index.isin(base.index)]
        obs = base.loc[per_gene.index, "logFC"].to_numpy()
        x = per_gene.to_numpy()
        r = stats.pearsonr(x, obs)[0] if len(x) >= 5 else np.nan
        null = np.array([stats.pearsonr(x, rng.permutation(obs))[0] for _ in range(2000)]) if len(x) >= 5 else np.array([np.nan])
        row = {"gene_set": label, "n_genes": len(per_gene), "pearson_r": r,
               "sign_agreement": float((np.sign(x) == np.sign(obs)).mean()) if len(x) else np.nan,
               "p_perm": float((1 + (null >= r).sum()) / 2001) if len(x) >= 5 else np.nan, "day_internal_h": t_hat}
        ee = per_gene.index.intersection(e_pred.index)          # partial r of ours given GSE88954's prediction
        if len(ee) >= 10:
            z = np.column_stack([np.ones(len(ee)), e_pred.loc[ee].to_numpy()])
            res = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
            row.update({"n_with_external_fit": len(ee),
                        "pearson_r_external_prediction": stats.pearsonr(e_pred.loc[ee], base.loc[ee, "logFC"])[0],
                        "partial_r_ours_given_external": stats.pearsonr(res(per_gene.loc[ee].to_numpy()),
                                                                        res(base.loc[ee, "logFC"].to_numpy()))[0]})
        tv.append(row)
    tv = pd.DataFrame(tv)
    write_csv(tv, MOD, "pilot_tierC_independent_validation.csv",
              "Tier C (selected without GSE107086) predicts the independent night − day difference (GSE107086, no noise, "
              "sham surgery)?", required=["gene_set", "n_genes", "pearson_r", "p_perm"])
    log("\n" + tv.round(3).to_string())

# ---- (B3) night programme in the independent cohort
npg = []
for k in ["night_minus_day_no_noise_shamsurg", "night_minus_day_no_noise_adx"]:
    d = de1[de1["contrast"] == k].set_index("gene")
    for name, gs in [("IEG", IEG), ("heat shock", HSG), ("NF-kB", NFKB), ("IL-1", IL1)]:
        x = d[d.index.isin(gs)]
        if len(x) < 3:
            continue
        npg.append({"contrast": k, "gene_set": name, "n_genes": len(x), "mean_logFC_night_minus_day": float(x["logFC"].mean()),
                    "frac_positive": float((x["logFC"] > 0).mean()), "n_p05_up": int(((x["p"] < 0.05) & (x["logFC"] > 0)).sum()),
                    "p_wilcoxon_vs_all": float(stats.mannwhitneyu(x["logFC"], d.loc[~d.index.isin(gs), "logFC"]).pvalue),
                    "genes_up": ";".join(x.index[(x["logFC"] > 0)])})
write_csv(pd.DataFrame(npg), MOD, "pilot_night_programme_check.csv",
          "GSE107086: night − day difference without noise (collection ~ZT17.5 vs ~ZT5.5) of the night activation programme "
          "genes, sham surgery and ADX", required=["contrast", "gene_set", "mean_logFC_night_minus_day", "p_wilcoxon_vs_all"])

# ---- (C) noise × time-of-day
tierc = set()
if tc_path.exists():
    t_ = pd.read_csv(tc_path, usecols=["gene", "tierC"])
    tierc = set(t_.loc[t_["tierC"].astype(str) == "True", "gene"])
night_genes = set(ours[(ours["p"] < 0.05) & ours["peak_zt"].between(12, 24, inclusive="left")].index)
day_genes = set(ours[(ours["p"] < 0.05) & (ours["peak_zt"] < 12)].index)
sets = {"IEG": set(IEG), "heat shock": set(HSG), "NF-kB": set(NFKB), "IL-1": set(IL1), "clock": CLOCKG,
        "our cochlea rhythmic, night peak": night_genes, "our cochlea rhythmic, day peak": day_genes,
        "our cochlea Tier C (calibrated, any type)": tierc}
ov = []
bg = set(base.index)
for k in ["noise_day_shamsurg", "noise_night_shamsurg", "noise_night_vs_day_shamsurg", "noise_night_vs_day_adx",
          "three_way_gc_dependence"]:
    d = de1[de1["contrast"] == k].set_index("gene")
    hit = set(d.index[d["p"] < 0.01])
    for name, s in sets.items():
        s = s & bg
        a = len(hit & s); b = len(hit - s); c = len(s - hit); dd = len(bg - hit - s)
        orr, p = stats.fisher_exact([[a, b], [c, dd]], alternative="greater") if len(s) else (np.nan, np.nan)
        ov.append({"contrast": k, "gene_set": name, "n_set_expressed": len(s), "n_hits": len(hit), "n_overlap": a,
                   "odds_ratio": orr, "p_fisher_greater": p,
                   "mean_logFC_in_set": float(d.loc[d.index.isin(s), "logFC"].mean()) if len(s) else np.nan})
ov = pd.DataFrame(ov)
ov["q_BH"] = stats.false_discovery_control(ov["p_fisher_greater"].fillna(1).to_numpy())
# does the baseline night − day difference (our prediction) predict the time-of-day dependence of the noise response?
g = ours[(ours["p"] < 0.05)]
inter = de1[de1["contrast"] == "noise_night_vs_day_shamsurg"].set_index("gene")
g = g[g.index.isin(inter.index)]
rho, p = stats.spearmanr(pred_delta(g, t_hat, O_C), inter.loc[g.index, "logFC"])
ov.loc[len(ov)] = {"contrast": "noise_night_vs_day_shamsurg", "gene_set": "Spearman: interaction logFC vs our predicted "
                   "baseline night − day difference (our rhythmic genes)", "n_set_expressed": len(g), "n_hits": np.nan,
                   "n_overlap": np.nan, "odds_ratio": rho, "p_fisher_greater": p, "mean_logFC_in_set": np.nan, "q_BH": np.nan}
write_csv(ov, MOD, "pilot_noise_geneset_overlap.csv",
          "GSE107086: overlap of noise / noise × time-of-day genes (p < 0.01) with atlas gene sets; last row = correlation "
          "with our predicted baseline night − day difference (odds_ratio column holds Spearman rho)",
          required=["contrast", "gene_set", "n_overlap", "odds_ratio", "p_fisher_greater"])

mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 2.5) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.4) & (mk["pct_nz_reference"] <= 0.1)]
sigs = {c: g_.sort_values("scores", ascending=False)["names"].head(25).tolist() for c, g_ in mk.groupby("cell_type")}
txt = gzip.open(GEO / "GSE107086/GSE107086_raw_counts.csv.gz", "rt", newline="").read().replace("\r", "\n")
cnt = pd.read_csv(io.StringIO(txt)).dropna(how="all").drop_duplicates("geneName").set_index("geneName")
cpm = np.log2(cnt / cnt.sum(0) * 1e6 + 1)
cpm = cpm[(cnt >= 10).sum(1) >= 3]
z = cpm.sub(cpm.mean(1), axis=0).div(cpm.std(1).replace(0, np.nan), axis=0)
meta = pd.DataFrame({"sample": cnt.columns})
meta["noise"] = np.where(meta["sample"].str.match(r"^Sh[DN]N"), "sham", "noise")
meta["tod"] = np.where(meta["sample"].str.match(r"^(Sh)?DN"), "day", "night")
meta["surgery"] = np.where(meta["sample"].str.contains("ShADX"), "sham", "ADX")
sg = []
for name, genes in sigs.items():
    gg = [x for x in genes if x in z.index]
    if len(gg) < 5:
        continue
    meta["score"] = z.loc[gg, meta["sample"]].mean(0).to_numpy()
    m = smf.ols("score ~ C(noise, Treatment('sham')) * C(tod, Treatment('day')) * C(surgery, Treatment('sham'))", meta).fit()
    pv = m.pvalues
    sg.append({"signature": name, "n_genes": len(gg),
               "noise_effect_day_shamsurg": m.params.iloc[1], "p_noise_day_shamsurg": pv.iloc[1],
               "noise_x_night_shamsurg": m.params.iloc[3], "p_noise_x_night_shamsurg": pv.iloc[3],
               "night_effect_no_noise": m.params.iloc[2], "p_night_effect_no_noise": pv.iloc[2],
               "three_way": m.params.iloc[-1], "p_three_way": pv.iloc[-1]})
sg = pd.DataFrame(sg)
for c in [c for c in sg.columns if c.startswith("p_")]:
    sg[c.replace("p_", "q_")] = stats.false_discovery_control(sg[c].to_numpy())
write_csv(sg, MOD, "pilot_noise_signatures.csv",
          "GSE107086: our cell-type signature scores, OLS score ~ noise × time-of-day × surgery (reference: sham noise, day, "
          "sham surgery)", required=["signature", "noise_effect_day_shamsurg", "p_noise_day_shamsurg"])

# ---- (D) kinetics after noise (GSE312476)
kin = []
for k, d in de2.groupby("contrast"):
    d = d.set_index("gene")
    for name, s in sets.items():
        x = d[d.index.isin(s)]
        if len(x) < 3:
            continue
        rest = d[~d.index.isin(s)]["logFC"]
        kin.append({"contrast": k, "gene_set": name, "n_genes": len(x), "mean_logFC": float(x["logFC"].mean()),
                    "frac_up_q05": float(((x["q_BH"] < 0.05) & (x["logFC"] > 0)).mean()),
                    "frac_down_q05": float(((x["q_BH"] < 0.05) & (x["logFC"] < 0)).mean()),
                    "p_wilcoxon_vs_rest": float(stats.mannwhitneyu(x["logFC"], rest).pvalue)})
write_csv(pd.DataFrame(kin), MOD, "pilot_noise_kinetics_genesets.csv",
          "GSE312476: noise-response kinetics (0/4/8 h after 2 h of 100 dB) of IEG / heat-shock / NF-κB / IL-1 sets and of "
          "our night- vs day-peaking cochlear genes", required=["contrast", "gene_set", "mean_logFC"])

# ---- (E) GeoMx spatial
gm = pd.read_csv(GEO / "GSE312646/GSE312646_GeoMX_table.csv.gz", header=0, low_memory=False)
gm = gm.loc[:, ~gm.columns.str.startswith("Unnamed")]
gm = gm.iloc[4:].dropna(subset=["TargetName"]).drop_duplicates("TargetName").set_index("TargetName")
gm = gm.apply(pd.to_numeric, errors="coerce").dropna()          # 65 genes carry a non-numeric flag ('x') in some AOIs
aoi = pd.DataFrame({"aoi": gm.columns})
aoi["region"] = np.where(aoi["aoi"].str.startswith("Neu"), "spiral_ganglion_neurons", "supporting_cells")
aoi["noise"] = np.where(aoi["aoi"].str.contains("_Nos_"), "noise", "control")
lg = np.log2(gm + 1)
zg = lg.sub(lg.mean(1), axis=0).div(lg.std(1).replace(0, np.nan), axis=0)
gs_rows = []
for name, genes in sigs.items():
    gg = [x for x in genes if x in zg.index]
    if len(gg) < 3:
        continue
    aoi["score"] = zg.loc[gg, aoi["aoi"]].mean(0).to_numpy()
    m = smf.ols("score ~ C(region) * C(noise, Treatment('control'))", aoi).fit()
    r_ = {"signature": name, "n_genes": len(gg)}
    for reg in ["spiral_ganglion_neurons", "supporting_cells"]:
        a = aoi[(aoi["region"] == reg) & (aoi["noise"] == "noise")]["score"]
        b = aoi[(aoi["region"] == reg) & (aoi["noise"] == "control")]["score"]
        t_, p_ = stats.ttest_ind(a, b, equal_var=False)
        r_[f"mean_control_{reg}"] = float(b.mean())
        r_[f"noise_effect_{reg}"] = float(a.mean() - b.mean())
        r_[f"p_noise_{reg}"] = float(p_)
    r_["neuron_minus_supporting_control"] = r_["mean_control_spiral_ganglion_neurons"] - r_["mean_control_supporting_cells"]
    r_["p_region_x_noise"] = float(m.pvalues.iloc[-1])
    gs_rows.append(r_)
gsx = pd.DataFrame(gs_rows)
for c in ["p_noise_spiral_ganglion_neurons", "p_noise_supporting_cells"]:
    gsx[c.replace("p_", "q_")] = stats.false_discovery_control(gsx[c].to_numpy())
write_csv(gsx, MOD, "pilot_geomx_signatures.csv",
          "GSE312646 GeoMx: our cell-type signature scores per region (localisation) and noise effects per region",
          required=["signature", "neuron_minus_supporting_control", "noise_effect_spiral_ganglion_neurons", "noise_effect_supporting_cells"])
rowsx = []
tcd = pd.read_csv(tc_path, usecols=["cell_type", "gene", "tierC"]) if tc_path.exists() else pd.DataFrame(columns=["cell_type", "gene", "tierC"])
tcd = tcd[tcd["tierC"].astype(str) == "True"]
sets_sp = dict(sets)
sets_sp["Tier C: Spiral Ganglion Neuron"] = set(tcd.loc[tcd["cell_type"] == "Spiral Ganglion Neuron", "gene"])
sets_sp["Tier C: epithelial / hair-cell types"] = set(tcd.loc[tcd["cell_type"].isin(
    ["Epithelial Cell (Non-sensory)", "Hair Cell (Outer/Inner)", "Interdental Cell"]), "gene"])
ratio = lg[aoi.loc[(aoi["region"] == "spiral_ganglion_neurons") & (aoi["noise"] == "control"), "aoi"]].mean(1) - \
    lg[aoi.loc[(aoi["region"] == "supporting_cells") & (aoi["noise"] == "control"), "aoi"]].mean(1)
for reg in ["spiral_ganglion_neurons", "supporting_cells"]:
    a = lg[aoi.loc[(aoi["region"] == reg) & (aoi["noise"] == "noise"), "aoi"]]
    b = lg[aoi.loc[(aoi["region"] == reg) & (aoi["noise"] == "control"), "aoi"]]
    tt = stats.ttest_ind(a, b, axis=1, equal_var=False)
    eff = a.mean(1) - b.mean(1)
    for name, s in sets_sp.items():
        idx = lg.index.isin(s)
        if idx.sum() < 3:
            continue
        rowsx.append({"region": reg, "gene_set": name, "n_genes": int(idx.sum()), "mean_noise_log2FC": float(eff[idx].mean()),
                      "frac_up_p05": float(((tt.pvalue < 0.05) & (eff > 0))[idx].mean()),
                      "p_wilcoxon_effect_vs_rest": float(stats.mannwhitneyu(eff[idx], eff[~idx]).pvalue),
                      "mean_neuron_minus_supporting_log2": float(ratio[idx].mean()),
                      "p_wilcoxon_localisation_vs_rest": float(stats.mannwhitneyu(ratio[idx], ratio[~idx]).pvalue)})
write_csv(pd.DataFrame(rowsx), MOD, "pilot_geomx_genesets.csv",
          "GSE312646 GeoMx: noise effect per region and neuron-vs-supporting localisation of gene sets (incl. our Tier C)",
          required=["region", "gene_set", "n_genes", "mean_noise_log2FC", "mean_neuron_minus_supporting_log2"])
log("done")
