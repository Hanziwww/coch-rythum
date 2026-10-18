"""Compile headline numbers from all modules into results/key_findings.csv (one row per finding, with the source
file it comes from). Missing inputs are skipped and reported as 'not available'."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

rows = []


def add(module, finding, value, source, note=""):
    rows.append({"module": module, "finding": finding, "value": value, "source": source, "note": note})


def safe(path, fn):
    p = RESULTS / path
    if not p.exists():
        add("-", f"{path} not available", "", path)
        return
    try:
        fn(pd.read_csv(p), path)
    except Exception as e:  # keep compiling
        add("-", f"{path} could not be summarised: {e}", "", path)


def clock(d, src):
    for t, g in d.groupby("tissue"):
        z = np.exp(1j * 2 * np.pi / 24 * g["clock_phase_offset_h"]) * g["phase_coherence"]
        add("M0 clock", f"{t}: pooled clock offset vs canonical ZT (h)", round(float(np.angle(z.mean()) * 24 / (2 * np.pi)), 2), src,
            "ZT = clock hour − 8; independent cohorts")


def ccd(d, src):
    for t, g in d.groupby("tissue"):
        add("M2 clock network", f"{t}: cell types with clock-like co-expression (CCD perm p < 0.05)",
            f"{int((g['Pvalue'] < 0.05).sum())}/{len(g)}", src, "; ".join(g.sort_values("z").head(6)["group"]))


def tiers(d, src):
    for t, g in d.groupby("tissue"):
        add("M2 gene rhythms", f"{t}: Tier-1 / Tier-2 rhythmic gene × cell-type calls",
            f"{int(g['n_tier1'].sum())} / {int(g['n_tier2'].sum())}", src,
            "top Tier-1: " + "; ".join(f"{r.group} ({r.n_tier1})" for r in g.sort_values("n_tier1", ascending=False).head(5).itertuples()))
        add("M2 gene rhythms", f"{t}: median pi1 (Storey) across cell types", round(float(g["pi1_storey"].median()), 3), src)


def dream(d, src):
    for r in d.itertuples():
        add("M2 shared/specific", f"{r.tissue}: genes with shared / cell-type-specific rhythm (q < 0.1)",
            f"{r.dream_shared_q10} / {r.dream_specific_q10}", src, "dream with sample random effect")


def decomposition(d, src):
    a = d[d["model"] == "marrow_adjusted"]
    add("M4 abundance", "cochlea: immune types with P(residual FC > 1.5) > 0.5 after marrow adjustment",
        int((a["prob_fc_gt_1p5"] > 0.5).sum()), src, "max P = " + str(round(float(a["prob_fc_gt_1p5"].max()), 2)))


def frames(d, src):
    r = d[(d["q_BH"] < 0.1) & d["robust"]]
    add("M4 abundance", "robust abundance rhythms (frame BH q < 0.1, perm p <= 0.05, LOTO stable)", len(r), src,
        "; ".join(f"{x.frame}:{x.tissue}:{x.feature}@ZT{x.peak_zt:.1f}" for x in r.itertuples()))


def axes(d, src):
    s = d[(d["binom_p_greater"] < 0.01)]
    add("M5 migration axes", "axis categories enriched for rhythmic potentials (binomial p < 0.01)", len(s), src,
        "; ".join(f"{x._1}:{x.category} ({x.enrichment_vs_5pct:.1f}x, ZT{x.circ_mean_peak_zt_sig:.1f})"
                  for x in s.rename(columns={"class": "_1"}).itertuples()))


def models(d, src):
    b = d.sort_values("weight", ascending=False).groupby("cell_type").head(1)
    add("M5 compartment models", "best-supported migration model per immune type (WAIC weight)",
        "; ".join(f"{r.cell_type}:{r.model}({r.weight:.2f})" for r in b.itertuples()), src,
        "M0 independent, M1 BM→cochlea, M2 shared driver, M3 cochlea→BM")


def external(d, src):
    for r in d.itertuples():
        add("M5 external", f"{r.dataset} {r.tissue} {r.condition}: samples / time points / reps",
            f"{r.n_samples} / {r.time_points} / {r.replicates_per_time}", src)


def ext_rhythm(path):
    p = RESULTS / path
    if p.exists():
        d = pd.read_csv(p)
        for (c), g in d.groupby("condition"):
            add("M5 external", f"{path.split('/')[-1]}: rhythmic genes (q < 0.05), {c}", int((g["q_BH"] < 0.05).sum()), path)


def tt(d, src):
    for r in d.itertuples():
        add("M0/M5 time-telling", f"{r.reference} → {r.target_tissue} {r.target}: circular offset (h)",
            round(r.circ_mean_offset_h, 2), src, f"resultant {r.offset_resultant:.2f}")


def meta(d, src):
    for r in d.itertuples():
        add("M5 meta-cosinor", f"{r.tissue} + {r.external_dataset}: genes q_meta < 0.05 (frac phase-concordant)",
            f"{r.n_q05} ({r.frac_concordant_among_q05:.2f})", src)


def depth_matched(d, src):
    for t, g in d.groupby("tissue"):
        pr, dm = g[g["version"] == "primary"], g[g["version"] == "depth_matched"]
        add("M2 sensitivity", f"{t}: mean q<0.1 genes per cell type, primary vs depth-matched (mean of 3 thinnings)",
            f"{pr['n_q10'].mean():.0f} vs {dm['n_q10'].mean():.0f}", src,
            f"Tier-1/2 genes keeping p<0.05 after depth matching: {dm['frac_tier12_p05'].mean():.2f}; "
            f"phase resultant of q<0.1 genes {pr['q10_phase_resultant'].mean():.2f} vs {dm['q10_phase_resultant'].mean():.2f}")


def statemix(d, src):
    for t, g in d.dropna(subset=["frac_tier12_p05_after"]).groupby("tissue"):
        add("M2 sensitivity", f"{t}: Tier-1/2 genes keeping p<0.05 after the within-type state-mix covariate (median over types)",
            round(float(g["frac_tier12_p05_after"].median()), 2), src,
            "lowest: " + "; ".join(f"{r.cell_type} ({r.frac_tier12_p05_after:.2f}, state~time R2 {r.state_time_r2:.2f})"
                                   for r in g.sort_values("frac_tier12_p05_after").head(3).itertuples()))


def glmm(d, src):
    for t, g in d.groupby("tissue"):
        t1 = g[g["tier"] == "tier1"]
        add("M2 sensitivity", f"{t}: Tier-1 calls with cell-level NB-GLMM p<0.05 (sample random intercept)",
            f"{(t1['p_glmm'] < 0.05).sum()}/{t1['p_glmm'].notna().sum()}", src,
            f"non-rhythmic clock genes p<0.05: {(g.loc[g['tier'] == 'none', 'p_glmm'] < 0.05).mean():.2f}; "
            f"sample SD estimated 0 in {(g['sample_sd'] < 1e-6).mean():.0%} of fits (time = sample, anticonservative)")


def lr(d, src):
    for t, g in d.groupby("tissue"):
        add("M2 independent methods", f"{t}: Tier-1 calls gained if the LR test (diffCircadian) were added",
            int(g["n_gained"].sum()), src, f"Tier-1 genes with LR p<0.05: {g['frac_tier1_lr_p05'].mean():.2f}")


def pathway_counts(d, src):
    d = d[d["resource"].isin(["hallmark", "gobp", "reactome", "progeny", "collectri"])]
    for (t, res), g in d.groupby(["tissue", "resource"]):
        v = g.set_index("version")
        add("M3 pathways", f"{t} {res}: robust pathway rhythms primary / depth-matched / within-sample-centred / both",
            " / ".join(str(int(v.loc[k, "n_robust"])) if k in v.index else "-"
                       for k in ["primary", "depth_matched", "centred", "centred_depth_matched"]), src,
            "share in modal 3-h phase bin: " + "; ".join(
                f"{k} {v.loc[k, 'modal_phase_bin_zt']} {v.loc[k, 'frac_robust_in_modal_bin']:.2f}"
                for k in v.index if v.loc[k, "n_robust"] > 0))


def tf_lag(d, src):
    for r in d.itertuples():
        add("M3 TF activity", f"{r.tissue} ({'clock TFs' if r.clock_tf else 'other TFs'}): TF activity minus mRNA peak lag",
            f"{r.circ_mean_lag_h:.1f} h (n={r.n_pairs})", src,
            f"resultant {r.resultant_length:.2f}, Rayleigh p {r.rayleigh_p:.3g}")


def motifs_tissue(d, src):
    for t, g in d.groupby("tissue"):
        s = g[g["q_BH"] < 0.1]
        add("M2/M3 motifs", f"{t}: promoter motifs enriched among rhythmic genes (CMH over cell type x expression, q<0.1)",
            "; ".join(f"{r.motif} OR {r.mh_odds_ratio:.2f}" for r in s.itertuples()) or "none", src)


def motifs_phase(d, src):
    s = d[d["q_directional_perm"] < 0.1]
    add("M2/M3 motifs", "motif carriers peaking at the expected phase (directional permutation q<0.1)", len(s), src,
        "; ".join(f"{r.tissue}:{r.cell_type}:{r.motif}@ZT{r.circ_mean_peak_zt:.1f}" for r in s.itertuples()))


def flux(d, src):
    n = d.set_index(["unit", "frame", "stage"])
    for key in [("neutrophil", "A_clr_all", "Neutrophil (Mature)"),
                ("neutrophil_postmitotic", "D_within_neutrophil", "Neutrophil (Mature)"),
                ("Neutrophil (Aged)", "D_within_neutrophil", "Neutrophil (Aged)"), ("Macrophage", "A_clr_all", "Macrophage"),
                ("B Cell (Naïve)", "D_within_B", "B Cell (Naïve)")]:
        if key in n.index:
            r = n.loc[key]
            add("M5 lineage flux", f"BM {key[2]} ({key[1]}): observed fold change vs largest residence time compatible with one gated rate",
                f"FC {r['obs_ptt_fc']:.2f}; tau_max {r['tau_max_exit_gating_h']:.1f} h (exit) / {r['tau_max_influx_gating_h']:.1f} h (influx)",
                src, f"literature-informed prior median {r['tau_prior_median_h']} h")


def flux_fits(d, src):
    for r in d[d["role"] == "return_or_recirculating_pool"].itertuples():
        add("M5 lineage flux", f"BM {r.stage} ({r.frame}): required influx rhythm (relative amplitude, peak ZT)",
            f"{r.rate_rel_amp_median:.2f} [{r.rate_rel_amp_q05:.2f}, {r.rate_rel_amp_q95:.2f}] @ZT{r.rate_peak_zt_circmean:.1f}",
            src, f"P(|M|>1) = {r.prob_infeasible_sinusoidal:.2f}")


def dependence(d, src):
    d = d.rename(columns={"class": "dep_class"})
    s = d[(d["q_BH"] < 0.1) & (~d["confident_only"]) & d["dep_class"].isin(["dependent", "independent", "gained"])]
    add("M5 external dependence", f"{d['dataset'].iloc[0]}: cell types whose rhythmic genes are enriched in a dependence class (q<0.1)",
        len(s), src, "; ".join(f"{r.tissue}:{r.cell_type}:{r.dep_class} OR {r.odds_ratio:.2f}" for r in s.itertuples()))


def gwas(d, src):
    s = d[d["cmh_q"] < 0.1]
    add("M6 GWAS", "rhythmic-gene overlap with GWAS trait genes after gene-length x expression stratification (CMH q<0.1)",
        len(s), src, "; ".join(f"{r.tissue}:{r.cell_type}:{r.trait_group} OR {r.cmh_or_len_expr:.2f}" for r in s.itertuples())
        + f" (unadjusted hypergeometric q<0.1: {(d['q_BH'] < 0.1).sum()})")


def abundance_power(d, src):
    g = d[(d["method"] == "propeller_moderated") & (d["alpha"] == 0.05) & (d["ptt_fc"] == 2)]
    for t, h in g.groupby("tissue"):
        add("M4 power", f"{t}: median power to detect a 2-fold peak-to-trough share rhythm (propeller, alpha 0.05)",
            round(float(h["power"].median()), 2), src,
            f"median minimal detectable FC (80 %): {h['min_fc_80pct_alpha05'].median():.2f}")


def cues(d, src):
    s = d[d["rayleigh_p"] < 0.05]
    add("M6 systemic cues", "cue gene sets with a common phase across cell types (Rayleigh p<0.05)", len(s), src,
        "; ".join(f"{r.tissue}:{r.set}@ZT{r.weighted_common_peak_zt:.1f}" for r in s.itertuples()))


def mash_summary(d, src):
    for r in d.itertuples():
        add("M2 shared/specific", f"{r.tissue}: mashr gene x cell-type rhythms with lfsr<0.05 (distinct genes)",
            f"{getattr(r, 'mash_sig_gene_celltype', 'NA')} ({getattr(r, 'mash_genes_sig_any', 'NA')})", src)


def ccc(path):
    p = RESULTS / path
    if p.exists():
        d = pd.read_csv(p, usecols=["q_BH", "mechanism"])
        add("M5 communication", f"{path.split('/')[-1]}: rhythmic interactions (q<0.1)", int((d["q_BH"] < 0.1).sum()), path,
            "mechanisms among q<0.1: " + "; ".join(f"{k} {v}" for k, v in d.loc[d["q_BH"] < 0.1, "mechanism"].value_counts().items()))


def tensor(d, src):
    s = d[d["p"] < 0.05]
    add("M5 communication", "Tensor-cell2cell factors with rhythmic sample loadings (p<0.05)", f"{len(s)}/{len(d)}", src,
        "; ".join(f"{r.tissue}:{r.factor}@ZT{r.peak_zt:.1f}" for r in s.itertuples()))


def bayesprism(d, src):
    for r in d.itertuples():
        if r.p_mean_shift < 0.05 or r.p_rhythm_difference < 0.05:
            add("M5 deconvolution", f"{r.dataset} {r.cell_type}: BayesPrism condition effect",
                f"mean-shift p {r.p_mean_shift:.3g}; rhythm-difference p {r.p_rhythm_difference:.3g}", src, r.mean_by_condition)


def timer(d, src):
    for r in d.itertuples():
        add("M5 neutrophil timer (GSE86619)", f"{r.tissue} {r.cell_type}: Spearman of predicted vs blood-neutrophil ZT13−ZT5 change",
            f"{r.spearman_all:.2f} (all {r.n_shared_genes} genes, perm p {r.p_perm_all:.3g}); {r.spearman_both_p05:.2f} "
            f"(n={r.n_both_p05} p<0.05 in both)", src,
            f"sign agreement {r.sign_agreement_both_p05:.2f}; without stress genes {r.spearman_all_no_stress:.2f}")


def timer_ko(d, src):
    s = d[d["q_BH"] < 0.1]
    add("M5 neutrophil timer (GSE86619)", "rhythmic neutrophil-lineage genes enriched among knockout-responsive genes (q<0.1)",
        len(s), src, "; ".join(f"{r.tissue}:{r.cell_type}:{r.external_contrast} OR {r.odds_ratio:.2f}" for r in s.itertuples()))


def lr_diff(d, src):
    for lab, g in d.groupby("model_label"):
        add("M2 cross-tissue", f"LR_diff cochlea vs BM, dryR class {lab}: genes with amplitude / phase difference (q<0.05)",
            f"{int((g['q_lr_amplitude'] < 0.05).sum())} / {int((g['q_lr_phase'] < 0.05).sum())} of {len(g)}", src)


def dryr_labelled(d, src):
    num = [c for c in d.columns if c.endswith("_bicw05")]
    add("M2 cross-tissue", "dryR genes per class with BIC weight >= 0.5 (summed over shared cell types)",
        "; ".join(f"{c.replace('_bicw05', '')}={int(d[c].sum())}" for c in num), src)


def simulation(d, src):
    g = d.groupby(["tissue", "rule", "ptt_fc_injected"]).agg(power=("power", "mean"), fdp=("fdp", "mean")).reset_index()
    for (t, rule), h in g.groupby(["tissue", "rule"]):
        add("Verification (simulation)", f"{t} {rule}: power / false-discovery proportion by injected fold change",
            "; ".join(f"FC{r.ptt_fc_injected:g}: {r.power:.2f}/{r.fdp:.2f}" for r in h.itertuples()), src,
            "background = time labels relabelled by a null dihedral class (structured, non-circadian variation)")


def rcistarget(d, src):
    d = d[d["set"] == "pooled"]
    for (t, b), g in d.groupby(["tissue", "phase_bin"]):
        g = g.sort_values("NES", ascending=False).head(4)
        add("M2/M3 motifs (RcisTarget)", f"{t} rhythmic genes peaking {b}: top enriched motifs (NES)",
            "; ".join(f"{r.motif} {r.NES:.1f}" for r in g.itertuples()), src,
            "TFs: " + " | ".join(str(x) for x in g["TF_highConf"].fillna("").tolist()))


def multinichenet(path):
    p = RESULTS / path
    if not p.exists():
        add("-", f"{path} not available", "", path)
        return
    d = pd.read_csv(p, usecols=["group", "sender", "receiver", "ligand", "receptor", "prioritization_score", "tissue"])
    for (t, grp), g in d.groupby(["tissue", "group"]):
        g = g.sort_values("prioritization_score", ascending=False).drop_duplicates(["ligand", "receptor"]).head(6)
        add("M5 MultiNicheNet", f"{t}: top prioritised ligand–receptor pairs up in the {grp} phase",
            "; ".join(f"{r.sender}→{r.receiver} {r.ligand}–{r.receptor}" for r in g.itertuples()), path,
            "rest = ZT3/7/11, active = ZT15/19/23 (3 vs 3 pooled samples)")


def milo(path):
    p = RESULTS / path
    if p.exists():
        d = pd.read_csv(p)
        s = d[d["spatial_fdr"] < 0.1]
        top = s.groupby("majority_cell_type").agg(n=("nhood", "size"), peak=("peak_zt", "median")).sort_values("n", ascending=False).head(6)
        add("M4 Milo", f"{d['tissue'].iloc[0]}: rhythmic neighbourhoods (spatial FDR < 0.1)", f"{len(s)}/{len(d)}", path,
            "; ".join(f"{k} n={r.n} @ZT{r.peak:.1f}" for k, r in top.iterrows()))


def bm_likeness(d, src):
    s = d[d["p"] < 0.05]
    add("M4/M5 BM-likeness", "cochlear immune cell types whose BM-likeness oscillates (p < 0.05)", f"{len(s)}/{len(d)}", src,
        "; ".join(f"{r.cell_type}@ZT{r.peak_zt:.1f} (perm p {r.p_perm:.2f})" for r in s.itertuples()))


def dr_selected(d, src):
    for fam, g in d.groupby("family"):
        g = g.sort_values("composite", ascending=False)
        ok = g[g["rejected_time_loss"] != True]
        add("M1 DR", f"{fam}: selected embedding (highest composite among non-rejected)",
            f"{ok['embedding'].iloc[0]} ({ok['composite'].iloc[0]:.3f})", src,
            "rejected for time-signal loss: " + "; ".join(g.loc[g["rejected_time_loss"] == True, "embedding"].head(6)))


def dryr_counts(d, src):
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    add("M2 cross-tissue", "dryR model counts (cochlea vs BM) summed over shared cell types",
        "; ".join(f"{c}={int(d[c].sum())}" for c in num), src)


safe("20_dr/dr_benchmark_summary.csv", dr_selected)
safe("02_clock_controls/clock_phase_order.csv", clock)
safe("34_clock_network/clock_ccd.csv", ccd)
safe("33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv", tiers)
safe("36_shared_specific/shared_specific_summary.csv", dream)
safe("52_marrow_decomposition/tissue_residual_rhythm.csv", decomposition)
safe("50_abundance/abundance_rhythm_frames.csv", frames)
safe("65_migration_axes/migration_axes_category_summary.csv", axes)
safe("60_migration/compartment_model_compare.csv", models)
safe("71_external/external_manifest.csv", external)
for f in sorted((RESULTS / "71_external").glob("external_rhythm_*.csv")):
    ext_rhythm(str(f.relative_to(RESULTS)))
safe("74_timetelling/timetelling_summary.csv", tt)
safe("73_meta/meta_cosinor_summary.csv", meta)
safe("33_rhythm_tiers/rhythm_pipeline_simulation.csv", simulation)
safe("39_sensitivity/depth_matched_summary.csv", depth_matched)
safe("39_sensitivity/statemix_summary.csv", statemix)
safe("38_cell_level/cell_level_glmm.csv", glmm)
safe("32_rhythm_alt/lr_tier_impact.csv", lr)
safe("36_shared_specific/shared_specific_summary.csv", mash_summary)
safe("35_crosstissue/crosstissue_dryR_labelled_counts.csv", dryr_labelled)
safe("35_crosstissue/crosstissue_lr_diff.csv", lr_diff)
safe("77_neutrophil_timer/neutrophil_timer_concordance.csv", timer)
safe("77_neutrophil_timer/neutrophil_timer_ko_enrichment.csv", timer_ko)
safe("43_pathway_phase/pathway_robust_counts.csv", pathway_counts)
safe("43_pathway_phase/tf_lag_summary.csv", tf_lag)
safe("44_motifs/motif_enrichment.csv", rcistarget)
safe("44_motifs/promoter_motif_enrichment_tissue.csv", motifs_tissue)
safe("44_motifs/promoter_motif_phase.csv", motifs_phase)
safe("54_abundance_power/abundance_power.csv", abundance_power)
safe("61_lineage_flux/lineage_flux_nonlinear.csv", flux)
safe("61_lineage_flux/lineage_flux_fits.csv", flux_fits)
safe("76_dependence/gc_dependence_mapping.csv", dependence)
safe("76_dependence/neutrophil_dependence_mapping.csv", dependence)
safe("84_gwas/gwas_enrichment.csv", gwas)
safe("82_systemic/systemic_cue_common_phase.csv", cues)
for f in sorted((RESULTS / "64_ccc_rhythm").glob("ccc_rhythm_*.csv")):
    ccc(str(f.relative_to(RESULTS)))
safe("64_ccc_rhythm/tensor_factor_rhythm.csv", tensor)
for t in ["bm", "cochlea"]:
    milo(f"53_milo/milo_da_results_{t}.csv")
safe("66_state_continua/bm_likeness_rhythm.csv", bm_likeness)
safe("75_deconvolution/bayesprism_condition_effects.csv", bayesprism)
for t in ["bm", "cochlea"]:
    multinichenet(f"67_multinichenet/multinichenet_prioritization_{t}.csv")
# ======================================================================= round 2 (ZT primary; clock = ZT + 8)
def r2_fig(d, src):
    add("R2 figure", "user-figure panels reproduced exactly (OLS cosinor, % of tissue cells, F(2,3))",
        f"{int(d['match'].sum())}/{len(d)}", src, "figure x-axis = clock time; true peaks: " +
        "; ".join(f"{r.tissue} {r.cell_type} ZT{r.true_peak_zt} (clock {r.replicated_peak_clock})" for r in d.itertuples()))


def r2_multi(d, src):
    for t, g in d.groupby("tissue"):
        sel = g[g["verdict"].isin(["robust", "supported"])].sort_values("verdict", key=lambda x: x.map({"robust": 0, "supported": 1}))
        add("R2 abundance", f"{t}: abundance-rhythm verdicts over 10 models",
            "; ".join(f"{k}={v}" for k, v in g["verdict"].value_counts().items()), src,
            "; ".join(f"{r.cell_type} {r.verdict} ZT{r.ols_pct_peak_zt:.1f} ({int(r.core_supporting)}/{int(r.core_tested)} core)"
                      for r in sel.itertuples()))


def r2_rel(d, src):
    b = d[d["both_p05"]]
    add("R2 cross-tissue", "cell types rhythmic (OLS-% p<0.05) in both tissues", len(b), src,
        "; ".join(f"{r.cell_type}: Δ(cochlea − BM) {r.delta_cochlea_minus_bm_h:+.1f} h [{r.delta_ci_lo:+.1f}, {r.delta_ci_hi:+.1f}]; "
                  f"P(emigration order) {r.prob_emigration_order:.2f}, P(return order) {r.prob_return_order:.2f}" for r in b.itertuples()))


def r2_pattern(d, src):
    for t, g in d[d["pc"] == 1].groupby("tissue"):
        r = g.iloc[0]
        add("R2 sample pattern", f"{t}: tissue-wide sample pattern PC1 (variance explained, 24-h cosinor)",
            f"{r.var_explained:.2f}; R² {r.pc_cosinor_r2:.2f}, p {r.pc_cosinor_p:.3f}, peak ZT{r.pc_cosinor_peak_zt:.1f}", src,
            f"largest single-sample share {r.single_sample_dominance:.2f}")


def r2_patcov(d, src):
    for t, g in d[d["pc"] == 1].drop_duplicates(["tissue", "covariate"]).groupby("tissue"):
        add("R2 sample pattern", f"{t}: PC1 correlation with sample covariates",
            "; ".join(f"{r.covariate} {r.pearson_r:+.2f}" for r in g.itertuples()), src)


def r2_tierc(d, src):
    sel = d[d["selected"].astype(str) == "True"]
    add("R2 Tier C", "rule ladder: calls (empirical FDR) per rule", "; ".join(
        f"{r.rule} {int(r.observed_calls)} ({r.empirical_fdr:.2f})" for r in d.itertuples()), src,
        f"selected (FDR target {d['fdr_target'].iloc[0]}): " + (sel["rule"].iloc[0] + " — " + sel["definition"].iloc[0] if len(sel) else "none"))


def r2_tierc_sim(d, src):
    s = d.groupby(["rule", "ptt_fc_injected"]).agg(power=("power", "mean"), fdp=("fdp", "mean")).reset_index()
    add("R2 Tier C", "simulation (injected rhythms): power / FDP by rule and FC", "; ".join(
        f"{r.rule} FC{r.ptt_fc_injected:g}: {r.power:.2f}/{r.fdp:.2f}" for r in s.itertuples()), src,
        "external / BM supports cannot be simulated (need real external rhythms)")


def r2_tierc_ct(d, src):
    add("R2 Tier C", "cell types with most calibrated Tier C calls (calls; empirical FDR)", "; ".join(
        f"{r.cell_type} {int(r.n_calls)} ({r.celltype_empirical_fdr:.2f})" for r in d.head(8).itertuples()), src)


def r2_offsets(d, src):
    tp = d[d["unit"] == "tissue_pool"]
    add("R2 phase", "clock offsets vs canonical (tissue pools; 95 % CI half-width)", "; ".join(
        f"{r.tissue} {r.offset_canonical_h:+.2f} ± {r.ci_halfwidth_h:.2f} h" for r in tp.itertuples()), src)
    b = d[d["unit"] == "bulk"]
    add("R2 phase", "external units with |offset| <= 2 h", f"{int((b['offset_canonical_h'].abs() <= 2).sum())}/{len(b)}", src,
        "outside: " + "; ".join(f"{r.dataset} {r.condition} {r.offset_canonical_h:+.1f}" for r in b[b["offset_canonical_h"].abs() > 2].itertuples()))


def r2_cov(d, src):
    s = d.groupby(["tissue", "ptt_fc_injected"])[["coverage_all", "coverage_determinate", "frac_determinate"]].mean().reset_index()
    add("R2 phase", "delta-method 95 % phase-CI coverage (all / determinate) and share determinate", "; ".join(
        f"{r.tissue} FC{r.ptt_fc_injected:g}: {r.coverage_all:.2f}/{r.coverage_determinate:.2f} ({r.frac_determinate:.2f})"
        for r in s.itertuples()), src)


def r2_conc(d, src):
    x = d[(d["genes"] == "non_clock")]
    m = x[((x["tissue"] == "bm") & x["external_unit"].str.startswith("GSE198770_bone_marrow|bone_marrow|control")) |
          ((x["tissue"] == "cochlea") & (x["external_unit"] == "GSE88954|cochlea|sham"))]
    for (t, st), g in m.groupby(["tissue", "our_set"]):
        ok = g[(g["p_perm"] < 0.05) & (g["frac_within_3h"] > g["null_frac_within_3h"])]
        add("R2 external", f"{t} ({st}): cell types whose non-clock gene phases agree with the matched external tissue (perm p<0.05)",
            f"{len(ok)}/{len(g)}", src, "; ".join(ok["cell_type"]))


def r2_dossier(d, src):
    for t, g in d.groupby("tissue"):
        add("R2 external", f"{t}: cell-type dossier conclusions", "; ".join(f"{k}={v}" for k, v in g["conclusion"].value_counts().items()), src)


def r2_lit(d, src):
    add("R2 literature", "literature phase comparisons (agrees / partly / disagrees)", "; ".join(
        f"{k}={v}" for k, v in d["agreement"].str.split(" ").str[0].value_counts().items()), src,
        "; ".join(f"{r.source} {r.tissue} {r.cell_type}: exp ZT{r.expected_peak_zt:.0f} vs ZT{r.our_peak_zt:.1f}" for r in d.itertuples()))


def r2_dep(d, src):
    n = d[d["signature"] == "Neutrophil lineage"]
    add("R2 external", "positive control: neutrophil-lineage signature change after depletion / KO (mean z difference, p)", "; ".join(
        f"{r.dataset} {r.tissue} {r.contrast.replace('_vs_control', '')}: {r.mean_diff:+.2f} (p {r.p:.3g})" for r in n.itertuples()), src)


def r2_skull(d, src):
    add("R2 skull marrow", "similarity of marrow regions to temporal-bone marrow (Spearman of composition)", "; ".join(
        f"{r.region} {r.spearman_vs_temporal_bone:.2f}" for r in d.sort_values("spearman_vs_temporal_bone", ascending=False).itertuples()), src)


def r2_lc(d, src):
    key = d[d["p_ols_pct_all_cells"] < 0.05]
    add("R2 audit", "abundance rhythms (OLS-% p<0.05) re-tested on high-confidence cells: keep p<0.05 / median |peak shift|",
        f"{int((key['p_ols_pct_high_confidence'] < 0.05).sum())}/{len(key)}; {np.nanmedian(np.abs(key['peak_shift_h'])):.2f} h", src)


def r2_dryr(d, src):
    o = d[d["run"] == "observed"].groupby("model_label")["N"].sum()
    n = d[d["run"] == "null"].groupby("model_label")["N"].sum() / 3
    add("R2 audit", "dryR classes, observed vs mean of null relabelings (3 cell types)", "; ".join(
        f"{k} {int(o.get(k, 0))} vs {n.get(k, 0):.0f}" for k in ["cochlea_only", "bm_only", "shared", "different"]), src)


def r2_scvi(d, src):
    add("R2 audit", "cochlear immune types with rhythmic BM-likeness on unsupervised scVI (p<0.05)", f"{int((d['p'] < 0.05).sum())}/{len(d)}", src,
        "scANVI: " + f"{int((d['scanvi_p'] < 0.05).sum())}/{len(d)}")


def r2_timer(d, src):
    x = d[d["genes"].isin(["all", "neutrophil_markers", "rhythmic_p05"]) & d["condition"].isin(["control", "sham"])]
    add("R2 audit", "neutrophil-timer replication: Spearman of predicted ZT13−ZT5 vs blood neutrophils (external controls)", "; ".join(
        f"{r.dataset} {r.genes} {r.spearman_rho:+.2f}" for r in x.itertuples()), src, "gene-level p ignore inter-gene correlation")


def r2_reg(d, src):
    add("R2 validation", "positive findings by verdict", "; ".join(f"{k}={v}" for k, v in d["verdict"].value_counts().items()), src,
        "; ".join(f"{r.finding_id} {r.verdict}" for r in d.itertuples()))


def r2_gap(d, src):
    add("R2 audit", "gap register status", "; ".join(f"{k}={v}" for k, v in d["status"].value_counts().items()), src)


def r2_feas(d, src):
    add("R2 feasibility", "treatment-response / spatial candidates by decision", "; ".join(
        f"{k}={v}" for k, v in d["decision"].value_counts().items()), src,
        "; ".join(f"{r.accession}: {r.decision}" for r in d.itertuples()))


def r2_pilot(d, src):
    for r in d[d["genes"] == "non-clock rhythmic (p<0.05)"].itertuples():
        add("R2 pilot", f"GSE107086 night − day (no noise) predicted by our cochlear rhythms, collection {r.collection_time}",
            f"r {r.pearson_r:.2f} (n {r.n_genes}, perm p {r.p_perm:.4f})", src, f"sign agreement {r.sign_agreement:.2f}")


def r2_ct(d, src):
    b = d[d["is_best"].astype(str) == "True"]
    add("R2 pilot", "GSE107086 day collection time inferred from clock / rhythmic genes (internal h)", "; ".join(
        f"{r.reference} [{r.genes}] {r.day_collection_internal_h:.1f} (r {r.pearson_r:.2f})" for r in b.itertuples()), src,
        "GEO characteristics ZT11/ZT23; design text ~ZT6/ZT18")


def r3_coch(d, src):
    im = d[d["immune"]]
    add("R3 cochlea", "cochlear immune share rhythms (unadjusted, robust/supported)", "; ".join(
        f"{r.cell_type} ZT{r.peak_zt:.1f}" for r in im[im["multimodel_verdict"].isin(["robust", "supported"])].sort_values("peak_zt").itertuples()), src,
        "myeloid shares peak in the rest phase, B-lineage shares in the active phase")


def r3_bmrel(d, src):
    add("R3 BM extension", "BM relaxed tier rule ladder: calls (empirical FDR)", "; ".join(
        f"{r.rule} {int(r.observed_calls)} ({r.empirical_fdr:.2f})" for r in d.itertuples()), src)


def r3_bmind(d, src):
    add("R3 BM extension", "held-out GSE198770 replicate 3: prediction r (perm p) by set", "; ".join(
        f"{r.gene_set} {r.pearson_r_stacked:+.3f} ({r.p_perm:.3f})" for r in d.itertuples()), src)


def r3_312298(d, src):
    x = d[d["collection_time"].str.startswith("design")]
    add("R3 response", "GSE312298 sorted BM neutrophils (PBS, ZT14 vs ZT22): predicted vs observed night − day", "; ".join(
        f"{r.predictor} r {r.pearson_r:+.3f} (p {r.p_perm:.4f})" for r in x.itertuples()), src)


def r3_handling(d, src):
    x = d.drop_duplicates("tissue")
    add("R3 audit", "dissociation-stress score: sample-term variance share and sample-effect peak", "; ".join(
        f"{r.tissue} {r.sample_term_variance_share:.2f}, ZT{r.sample_effect_peak_zt:.1f}" for r in x.itertuples()), src)


def r3_cellclock(d, src):
    add("R3 audit", "cell types whose cell-level clock phase tracks ZT (exact p <= 0.05)", f"{int((d['p_perm'] <= 0.05).sum())}/{len(d)}", src,
        "; ".join(d.loc[d["p_perm"] <= 0.05, "cell_type"]))


def r3_sub(d, src):
    add("R3 audit", "label-free immune sub-states with rhythmic within-type share (perm p <= 0.05 / q < 0.1)",
        f"{int((d['p_perm'] <= 0.05).sum())} / {int((d['q_BH'] < 0.1).sum())} of {len(d)}", src)


def r3_geomx(d, src):
    s_ = d.set_index("cell_type")
    add("R3 spatial", "GeoMx deconvolution: SGN share neuron vs supporting AOIs; macrophage share",
        f"SGN {s_.loc['Spiral Ganglion Neuron', 'mean_prop_neuron_control']:.2f} vs {s_.loc['Spiral Ganglion Neuron', 'mean_prop_supporting_control']:.2f}; "
        f"macrophage {s_.loc['Macrophage', 'mean_prop_neuron_control']:.3f} vs {s_.loc['Macrophage', 'mean_prop_supporting_control']:.3f}", src)


def r3_noise(d, src):
    x = d[(d["subset"] == "all cells") & d["label"].isin(["Neutrophil (Mature)", "Neutrophil (Aged)", "Macrophage"])]
    add("R3 response", f"{src.split('/')[0]}: noise-induced share changes (log2)", "; ".join(
        f"{r.contrast} {r.label} {r.log2_ratio:+.2f}" for r in x.itertuples()), src)


def r4_bench(d, src):
    a = d[(d["unit"] == "ALL") & d["rule"].isin(["tier1", "tier2", "nominal"])]
    for (ds, cond), g in a.groupby(["dataset", "condition"]):
        add("R4 benchmark", f"{ds} [{cond}]: calls / realised FDP (held-out truth p > 0.2) / sensitivity", "; ".join(
            f"{r.rule} {r.mean_calls:.0f} / {r.mean_fdp_strict:.3f} / {r.mean_sensitivity:.3f}" for r in g.itertuples()), src)


def r4_attr(d, src):
    for t, g in d.groupby("tissue"):
        add("R4 attribution", f"{t}: replicated bulk rhythms reproduced by composition / regulation (share vs null)", "; ".join(
            f"{r['class']} {r.share:.3f} vs {r.null_share:.3f} (p {r.p_perm:.3f})" for _, r in g.iterrows()), src)


def r4_attr_enr(d, src):
    x = d[(d["subset"] == "all rhythmic") & (d["p_enriched"] < 0.05)].sort_values("ratio", ascending=False)
    for t, g in x.groupby("tissue"):
        add("R4 attribution", f"{t}: cell types enriched among replicated bulk rhythmic genes", "; ".join(
            f"{r.cell_type} ×{r.ratio:.2f}" for r in g.head(8).itertuples()), src)


def r4_drv(d, src):
    x = d[d["cell_type"] == "ALL"]
    for (t, ts), g in x.groupby(["tissue", "tier_set"]):
        add("R4 drivers", f"{t} {ts}: primary driver shares", "; ".join(f"{r.driver} {r.share:.2f}" for r in g.itertuples()), src)


def r4_dn(d, src):
    add("R4 drivers", "constant-darkness phase support (observed vs phase-permutation null)", "; ".join(
        f"{r.finding_id} {r.frac_dd_support:.2f} vs {r.frac_dd_support_phase_null:.2f}" for r in d.dropna(subset=["frac_dd_support"]).itertuples()), src)


def r4_bab(d, src):
    x = d[(d["genes"] == "non_clock") & d["our_scope"].isin(["ALL (Tier 1/2)", "Tier 1"])]
    add("R4 cross-species", "baboon (clock-aligned) phase concordance of BM Tier-1/2 genes: within 3 h vs null (p)", "; ".join(
        f"{r.baboon_tissue} {r.our_scope} {r.frac_within_3h:.2f} vs {r.null_frac_within_3h:.2f} ({r.p_perm:.3f})" for r in x.itertuples()), src)


def r4_hs(d, src):
    x = d[d["baboon_tissue"].str.startswith("SUMMARY")]
    add("R4 cross-species", "heat-shock programme peak: baboon raw ZT / aligned vs mouse aligned", "; ".join(
        f"{r.baboon_tissue.replace('SUMMARY ', '')}: ZT {r.peak_zt:.1f} / aligned {r.aligned:.1f}" for r in x.itertuples()), src)


def r4_hum(d, src):
    x = d[(d["primary"].astype(str) == "True")]
    add("R4 cross-species", "human blood neutrophils 14:00 − 08:00 vs our neutrophil-state predictions (r, perm p)", "; ".join(
        f"{r.tissue} {r.cell_type} {r.pearson_r:+.3f} ({r.p_perm:.4f})" for r in x.itertuples()), src)


def r4_conc(d, src):
    add("R4 concordance", "external adult cochlear atlases: confident share / types recovered / marker pass", "; ".join(
        f"{r.key} {r.frac_confident:.2f} / {int(r.n_types_recovered_20_confident)} / {r.marker_check_pass_rate:.2f}" for r in d.itertuples()), src)


def r4_auth(d, src):
    t = d[d["author_celltype"].str.startswith("ALL")]
    if len(t):
        add("R4 concordance", "GSE234926 author labels in the expected gold-standard label (cell-weighted); ARI",
            f"{t['frac_expected'].iloc[0]:.2f}; ARI {t['dominant_share'].iloc[0]:.2f}", src)


def r4_age(d, src):
    x = d.groupby(["contrast", "gene_set"])["ratio"].median().reset_index()
    add("R4 ageing", "median |log2FC| ratio rhythmic / matched background (per label, median)", "; ".join(
        f"{r.contrast} {r.gene_set} {r.ratio:.2f}" for r in x.itertuples()), src)


for path, fn in [("r4_benchmark/benchmark_summary.csv", r4_bench), ("r4_attribution/attribution_mechanism_summary.csv", r4_attr),
                 ("r4_attribution/attribution_celltype_enrichment.csv", r4_attr_enr), ("r4_drivers/driver_landscape.csv", r4_drv),
                 ("r4_drivers/diurnal_circadian_status.csv", r4_dn), ("r4_crossspecies/baboon_bm_gene_concordance.csv", r4_bab),
                 ("r4_crossspecies/baboon_heatshock_phase.csv", r4_hs), ("r4_crossspecies/human_neutrophil_timeofday.csv", r4_hum),
                 ("r4_concordance/atlas_concordance_summary.csv", r4_conc), ("r4_concordance/author_label_agreement.csv", r4_auth),
                 ("r4_ageing/ageing_rhythmic_variability.csv", r4_age)]:
    if (RESULTS / path).exists():
        safe(path, fn)

def r5_prim(d, src):
    add("R5 B cells", "primary hypotheses (decision D34: BH q across 12 tests + >= 2 agreeing methods)", "; ".join(
        f"{r.hypothesis.split(' ')[0]} {r.tissue}: {r.decision} (p {r.primary_p:.3f}, q {r.q_BH_primary_set:.2f}, methods {int(r.n_methods_agree)}/{int(r.n_methods)})"
        for r in d.itertuples()), src)


def r5_wave(d, src):
    x = d[d["selection"] == "nominal p < 0.1"]
    add("R5 B cells", "label-free B developmental wave (Milo neighbourhood peak vs scVI pseudotime)", "; ".join(
        f"{r.tissue} r {r.circ_linear_r:.2f} p {r.p_perm:.4f} [{r.peak_by_pseudotime_quintile}]" for r in x.itertuples()), src)


def r5_imm(d, src):
    b = d[(d["is_best"].astype(str) == "True") & (d["reference"].str.contains("microarray"))]
    add("R5 B cells", "ImmGen identity of our B labels (best-matching sorted fraction, microarray)", "; ".join(
        f"{r.tissue} {r.our_stage.split(' (')[0] if 'Cell' in r.our_stage.split(' (')[0] else r.our_stage} → {r.immgen_population}" for r in b.itertuples()), src)


def r5_ext(d, src):
    x = d[(d["dataset"] == "GSE198770 BM bulk") & d["signature"].isin(["ImmGen B.FrF.BM", "ImmGen B.FrE.BM"]) & d["peak_our_bm_zt_equiv"].notna()]
    add("R5 B cells", "GSE198770 BM bulk: ImmGen FrF / FrE signature peaks (our BM ZT equivalent)", "; ".join(
        f"{r.condition} {r.signature.replace('ImmGen ', '')} ZT{r.peak_our_bm_zt_equiv:.1f} (p {r.p:.4f})" for r in x.itertuples()), src)


def r5_ko(d, src):
    x = d[d["gene_or_set"] == "all B stages"]
    add("R5 B cells", "IgA-PC ZT12 − ZT0 predicted by our B rhythms: WT vs B-cell Bmal1 KO (r)", "; ".join(
        f"{r.analysis.replace('our ', '').replace(' rhythms predict IgA-PC ZT12 − ZT0', '')}: {r.WT_value if pd.notna(r.WT_value) else r.KO_value:+.3f}"
        for r in x.itertuples() if 'predict' in r.analysis), src)


def r5_clock(d, src):
    x = d[d["unit"].isin(["B lineage (pooled)", "T Cell (CD3+)", "NK Cell"])]
    add("R5 B cells", "clock offset vs tissue pool (gene bootstrap; h ± 95 % half-width)", "; ".join(
        f"{r.tissue} {r.unit} {r.diff_vs_tissue_pool_h:+.1f} ± {r.diff_ci_halfwidth_h:.1f}" for r in x.itertuples() if pd.notna(r.diff_vs_tissue_pool_h)), src)


for path, fn in [("r5_bcell/primary_hypotheses.csv", r5_prim), ("r5_bcell/b_milo_wave.csv", r5_wave), ("r5_bcell/immgen_label_mapping.csv", r5_imm),
                 ("r5_bcell/b_external_signature_rhythm.csv", r5_ext), ("r5_bcell/b_external_bmal1ko.csv", r5_ko), ("r5_bcell/b_clock_offsets.csv", r5_clock)]:
    if (RESULTS / path).exists():
        safe(path, fn)


def r6_prim(d, src):
    x = d[d["decision"] != "pending"]
    add("R6 myeloid", "primary hypotheses (decision D45 / D48: Fisher across tissues, BH q across 8 + >= 2 agreeing methods in one tissue + direction)", "; ".join(
        f"{r.hypothesis}: {r.decision} (p {r.combined_p:.3f}, q {r.q_BH:.2f}, methods {int(r.n_agreeing_methods_best_tissue)}, direction {r.direction_ok})"
        for r in x.itertuples()), src)


def r6_m2(d, src):
    x = d[(d["axis"] == "blood timer ZT13−ZT5 (GSE86619)") & (d["group"] == "pooled neutrophils") & (d["subset"] == "all cells")]
    add("R6 myeloid", "blood-timer (GSE86619 ZT13 − ZT5) score in pooled neutrophils: per-tissue peaks (exact p)", "; ".join(
        f"{r.tissue} ZT{r.peak_zt:.1f} (p {r.p_perm:.3f})" for r in x.itertuples()), src)


def r6_comp(d, src):
    x = d[d["frame"].isin(["within_neutrophil", "within_mono_mac"]) & d["verdict"].isin(["robust", "supported"])]
    add("R6 myeloid", "within-lineage myeloid composition rhythms (robust / supported)", "; ".join(
        f"{r.tissue} {r.feature} {r.verdict} ZT{r.peak_zt:.1f}" for r in x.itertuples()), src)


def r6_milo(d, src):
    x = d[d["selection"] == "all neutrophil-lineage neighbourhoods"]
    add("R6 myeloid", "label-free neutrophil-lineage wave (Milo neighbourhood peak vs scVI pseudotime)", "; ".join(
        f"{r.tissue} r {r.circ_linear_r:.2f} p {r.p_perm:.4f} [{r.peak_by_pseudotime_quintile}]" for r in x.itertuples()), src)


def r6_ratio(d, src):
    x = d[(d["score"] == "cxcr4_cxcr2") & (d["subset"] == "all cells")]
    add("R6 myeloid", "neutrophil Cxcr4 / Cxcr2 log-ratio (per-sample mean): peak and exact p", "; ".join(
        f"{r.tissue} ZT{r.peak_zt:.1f} (p {r.p_perm:.3f})" for r in x.itertuples()), src)


def r6_clock(d, src):
    add("R6 myeloid", "cell-level clock tracking of myeloid groups (offset resultant, exact p; usable cells)", "; ".join(
        f"{r.tissue} {r.group}: R {r.offset_resultant:.2f} p {r.p_perm:.3f} (usable {r.frac_cells_usable:.2f})" for r in d.itertuples() if pd.notna(r.offset_resultant)), src)


for path, fn in [("r6_myeloid/m_primary_hypotheses.csv", r6_prim), ("r6_myeloid/m_axes_rhythm.csv", r6_m2), ("r6_myeloid/m_composition_summary.csv", r6_comp),
                 ("r6_myeloid/m_milo_wave.csv", r6_milo), ("r6_myeloid/m_niche_ratio_rhythm.csv", r6_ratio), ("r6_myeloid/m_clock_cell_level.csv", r6_clock)]:
    if (RESULTS / path).exists():
        safe(path, fn)


def r9_flow(d, src):
    for r in d[d["family"].isin(["primary", "secondary"]) & d["verdict"].notna()].itertuples():
        add("R9 flow cohort", f"{r.test} {r.tissue.replace('_', ' ')}: {r.readout}, {r.effect_scale} (predicted {r.prediction})",
            f"{r.effect:.2f} ({r.ci_lo:.2f}–{r.ci_hi:.2f})", src, f"{r.verdict}; p {r.p:.2g}" +
            (f", Holm {r.p_holm:.2g}" if pd.notna(r.p_holm) else "") + f"; n {int(r.n_zt5)} + {int(r.n_zt17)} mice; register {r.finding}")


if (RESULTS / "r9_flow/flow_tests.csv").exists():
    safe("r9_flow/flow_tests.csv", r9_flow)

for path, fn in [("r3_cochlea/cochlea_celltype_oscillations.csv", r3_coch), ("r3_bm/bm_relaxed_rule_selection.csv", r3_bmrel),
                 ("r3_bm/bm_independence_validation.csv", r3_bmind), ("r3_response/gse312298_validation.csv", r3_312298),
                 ("r3_audit/handling_sample_effect.csv", r3_handling), ("r3_audit/cell_clock_phase_summary.csv", r3_cellclock),
                 ("r3_audit/immune_substates.csv", r3_sub), ("r3_spatial/geomx_deconvolution_summary.csv", r3_geomx),
                 ("r3_gse167078/composition_contrasts.csv", r3_noise), ("r3_gse346691/composition_contrasts.csv", r3_noise),
                 ("r3_ma2025/composition_contrasts.csv", r3_noise)]:
    if (RESULTS / path).exists():
        safe(path, fn)

for path, fn in [("r2_figure/figure_replication_check.csv", r2_fig), ("r2_figure/abundance_multimodel_summary.csv", r2_multi),
                 ("r2_figure/figure_phase_relationships.csv", r2_rel), ("r2_cochlea/sample_pattern_scores.csv", r2_pattern),
                 ("r2_cochlea/sample_pattern_covariates.csv", r2_patcov), ("r2_cochlea/cochlea_relaxed_rule_selection.csv", r2_tierc),
                 ("r2_cochlea/cochlea_relaxed_simulation.csv", r2_tierc_sim), ("r2_cochlea/tierC_celltype_summary.csv", r2_tierc_ct),
                 ("r2_phase/clock_alignment_offsets.csv", r2_offsets), ("r2_phase/phase_ci_coverage.csv", r2_cov),
                 ("r2_external/gene_phase_concordance.csv", r2_conc), ("r2_external/celltype_validation_dossier.csv", r2_dossier),
                 ("r2_external/literature_comparison.csv", r2_lit), ("r2_external/signature_depletion_effect.csv", r2_dep),
                 ("r2_external/skull_marrow_similarity.csv", r2_skull), ("r2_audit/label_confidence_abundance.csv", r2_lc),
                 ("r2_audit/dryr_null_check.csv", r2_dryr), ("r2_audit/bm_likeness_scvi_rhythm.csv", r2_scvi),
                 ("r2_audit/neutrophil_timer_replication.csv", r2_timer), ("r2_audit/positive_findings_register.csv", r2_reg),
                 ("r2_audit/gap_register.csv", r2_gap), ("r2_feasibility/feasibility_treatment_spatial.csv", r2_feas),
                 ("r2_pilots/pilot_daynight_rhythm_validation.csv", r2_pilot), ("r2_pilots/pilot_collection_time_inference.csv", r2_ct)]:
    safe(path, fn)

write_csv(pd.DataFrame(rows), ".", "key_findings.csv", "Headline numbers per module with source files",
          required=["module", "finding", "value", "source"])
log(f"{len(rows)} findings compiled")
