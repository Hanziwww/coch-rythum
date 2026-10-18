"""Round 3, WP7 — concluding synthesis of the whole analysis (rounds 1–3), built only from result tables.

(1) r3_conclusions.csv: one row per conclusion, grouped by the user's questions (figure / cochlear oscillations, flexible
    criteria, external validation, phase, audit, response and spatial data, BM extension), with verdict, the key numbers
    (pulled from the result files), the evidence files and the figure-ready tables. Rows that depend on the Ma 2025 data
    are marked 'pending' until results/r3_ma2025/ exists.
(2) noise_response_timecourse.csv: noise-response summaries across public datasets ordered by time after exposure
    (GSE107086 2 h; GSE312476 0–8 h; GSE167078 3 / 7 / 14 d; GSE346691 3 d; Ma 2025 3 d when available), for immune
    shares (single-cell, our labels) and gene programmes (bulk).
Round 4 adds sections 6 (design / atlas credibility) and 7 (drivers and conservation); round 5 adds section 8 (B cells);
round 6 adds section 9 (myeloid cells); round 7 adds section 12 (exploratory reassessment, 680); round 8 adds section 13
(Ma 2025 × atlas integration, 720); round 9 adds the independent flow-cytometry cohort (932) to section 3.
Outputs (results/): r3_conclusions.csv; results/r3_response/noise_response_timecourse.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402


def rd(rel, **kw):
    p = RESULTS / rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


reg = rd("r2_audit/positive_findings_register.csv").set_index("finding_id")


def v(fid):
    return reg["verdict"].get(fid, "not available") if len(reg) else "not available"


rows = []


def add(section, claim, verdict, numbers, evidence, figure_tables=""):
    rows.append({"section": section, "claim": claim, "verdict": verdict, "key_numbers": numbers, "evidence_files": evidence,
                 "figure_ready_tables": figure_tables})


# ---- 1. figure and cochlear oscillations
fr = rd("r2_figure/figure_replication_check.csv")
add("1 figure / cochlear oscillations", "The user figure is reproduced exactly; its x-axis is clock time (true ZT = clock − 8)",
    "confirmed", f"{int(fr['match'].sum()) if len(fr) else 0}/6 panels reproduced", "r2_figure/figure_replication_check.csv",
    "r2_figure/figure_curves.csv; r2_figure/figure_points.csv")
co = rd("r3_cochlea/cochlea_celltype_oscillations.csv")
if len(co):
    im = co[co["immune"]]
    my = im[im["multimodel_verdict"].isin(["robust", "supported"]) & (im["peak_zt"] < 12)]
    ly = im[im["multimodel_verdict"].isin(["robust", "supported"]) & (im["peak_zt"] >= 12)]
    rr = im.set_index("cell_type")["r_with_sample_pattern_pc1"]
    add("1 figure / cochlear oscillations", "Cochlear immune composition alternates: myeloid shares peak in the rest phase, B-lineage "
        "shares in the active phase (unadjusted shares; BM as sole source). Shares are compositionally coupled and both arms "
        f"follow the marrow-content axis (r with PC1: myeloid {rr.reindex(my['cell_type']).median():+.2f}, B-lineage "
        f"{rr.reindex(ly['cell_type']).median():+.2f})", "descriptive (compositional; per-type verdicts in the register)",
        "rest-phase: " + ", ".join(f"{r.cell_type} ZT{r.peak_zt:.1f}" for r in my.itertuples()) + " | active-phase: " +
        ", ".join(f"{r.cell_type} ZT{r.peak_zt:.1f}" for r in ly.itertuples()),
        "r3_cochlea/cochlea_celltype_oscillations.csv", "r3_cochlea/cochlea_schematic_peaks.csv; r3_cochlea/cochlea_immune_curves.csv")
add("1 figure / cochlear oscillations", "BM⇄cochlea emigration / return schematic", v("X01"),
    "replaced by the cochlea-only schematic (user decision)", "r2_audit/validation_matrix.csv (X01)", "r3_cochlea/cochlea_schematic_peaks.csv")
for fid, lab in [("A05", "cochlear band neutrophils ZT5.2"), ("A04", "cochlear classical monocytes ZT7.0"), ("A06", "cochlear mature neutrophils ZT8.6"),
                 ("A01", "BM naïve B ZT20.5"), ("A03", "BM transitional B ZT12.8"), ("A02", "BM classical monocytes ZT12.7")]:
    add("1 figure / cochlear oscillations", f"Figure panel: {lab}", v(fid),
        f"evidence dimensions supporting {reg['n_supports'].get(fid, np.nan)}, against {reg['n_against'].get(fid, np.nan)}" if len(reg) else "",
        "r2_audit/validation_matrix.csv")

# ---- 2. flexible cochlear criteria
sel = rd("r2_cochlea/cochlea_relaxed_rule_selection.csv")
tv = rd("r2_pilots/pilot_tierC_independent_validation.csv")
if len(sel):
    add("2 flexible cochlear criteria", "A relaxed cochlear gene × cell-type tier cannot be calibrated below FDR 0.2", v("G06"),
        "; ".join(f"{r.rule} {int(r.observed_calls)} (FDR {r.empirical_fdr:.2f})" for r in sel.itertuples()) +
        (f"; independent cohort (BM-support-only calls) r {tv.set_index('gene_set').iloc[1]['pearson_r']:+.3f}" if len(tv) > 1 else ""),
        "r2_cochlea/cochlea_relaxed_*.csv; r2_pilots/pilot_tierC_independent_validation.csv", "r2_cochlea/tierC_catalogue.csv")
cc = rd("r3_audit/cell_clock_phase_summary.csv")
add("2 flexible cochlear criteria", "Reliable cochlear rhythm evidence = structural-cell clocks (gene and cell level) and tissue-level rhythms",
    f"{v('G04')} / {v('R04')}", (f"cell-level clock tracks ZT in {int(((cc['tissue'] == 'cochlea') & (cc['p_perm'] <= 0.05)).sum())} cochlear types"
                                 if len(cc) else ""), "34_clock_network; r3_audit/cell_clock_phase_summary.csv")

# ---- 3. external validation and prior knowledge
add("3 external validation", "BM neutrophil-state rhythms replicate in sorted BM neutrophils of an independent study (GSE312298)", v("R01"),
    "", "r3_response/gse312298_validation.csv")
add("3 external validation", "Night heat-shock programme is physiological; the immediate-early / NF-κB night component tracks sample handling",
    v("R05"), "", "r3_audit/handling_*.csv; r2_pilots/pilot_night_programme_check.csv")
add("3 external validation", "Cochlear neutrophils follow the blood-neutrophil ZT13−ZT5 programme, temporal-BM neutrophils oppose it", v("G02"),
    "", "77_neutrophil_timer; r2_audit/neutrophil_timer_replication.csv")
add("3 external validation", "Core clocks of both tissues are entrained to the light cycle", v("G09"), "", "r2_phase/clock_alignment_offsets.csv")
fl = rd("r9_flow/flow_tests.csv")
if len(fl):                         # round 9 (D89): independent flow-cytometry cohort, pre-specified (932); register line of A05 / A04 / A01 / G02
    pr_ = fl[fl["family"] == "primary"]
    sc_ = fl[(fl["family"] == "secondary") & (fl["tissue"] == "cochlea")]
    rep_ = list(pr_.loc[pr_["verdict"] == "replicated", "test"]) + [t for t, g in sc_.groupby("test") if (g["verdict"] == "replicated").all()]
    nrep_ = [t for t in list(pr_["test"]) + list(sc_["test"].unique()) if t not in rep_]
    add("3 external validation", "An independent flow-cytometry cohort (20 mice, ZT5 vs ZT17; pre-specified) reproduces the predicted day–night "
        "differences: cochlear immature neutrophils and classical monocytes lower at ZT17, marrow recirculating B higher; CXCR4 on cochlear "
        "neutrophils lower at ZT17 with marrow flat", f"replicated {len(rep_)}/{len(rep_) + len(nrep_)} ({', '.join(rep_)})" +
        (f"; not replicated: {', '.join(nrep_)}" if nrep_ else ""),
        "; ".join(f"{r.test} ZT17 / ZT5 {r.effect:.2f} ({r.ci_lo:.2f}–{r.ci_hi:.2f}), Holm p {r.p_holm:.2g}" for r in pr_.itertuples()) + " | " +
        "; ".join(f"{r.test} {r.readout} cochlea {r.effect:.2f} (p {r.p:.2g}): {r.verdict}" for r in sc_.itertuples()) +
        " | register line 'independent_flow_cohort' of " + ", ".join(sorted(fl["finding"].dropna().unique())),
        "r9_flow/flow_tests.csv; r2_audit/validation_matrix.csv", "r9_figures/f4f_flow_shares.csv; r9_figures/f5i_flow_markers.csv")

# ---- 4. BM extension
bi = rd("r3_bm/bm_independence_validation.csv")
bsel = rd("r3_bm/bm_relaxed_rule_selection.csv")
if len(bsel):
    add("4 BM extension", "The cochlear relaxed-tier methodology applied to BM", v("R02"),
        "; ".join(f"{r.rule} {int(r.observed_calls)} (FDR {r.empirical_fdr:.2f})" for r in bsel.itertuples()),
        "r3_bm/bm_relaxed_*.csv", "")
if len(bi):
    b = bi.set_index("gene_set")
    add("4 BM extension", "BM Tier 1 (strict) remains the recommended set; it predicts held-out external BM", v("R03"),
        f"Tier 1 r {b.loc['BM Tier 1', 'pearson_r_stacked']:+.3f} (p {b.loc['BM Tier 1', 'p_perm']:.4f}); external rep 1–2 "
        f"{b.loc['BM Tier 1', 'pearson_r_external_rep12_prediction']:+.3f}", "r3_bm/bm_independence_validation.csv")

# ---- 5. response and spatial data
add("5 response / spatial data", "Noise expands cochlear neutrophils and macrophages within days", v("R06"), "", "r3_gse167078; r3_gse346691")
add("5 response / spatial data", "Atlas cell profiles localise correctly in public cochlear GeoMx data", v("R07"), "", "r3_spatial/*")
ma = (RESULTS / "r3_ma2025").exists()
add("5 response / spatial data", "Ma 2025: night- vs day-noise responses relate to the cochlear rhythms at the exposure times",
    v("R08") if ma else "pending (Ma 2025 matrices to be uploaded by the user)", "", "r3_ma2025/* (pipeline dry-run passed: r3_ingest/ma2025_pipeline_dryrun.csv)")
add("5 response / spatial data", "No adult cochlea / temporal-bone / skull-marrow spatial dataset beyond GSE312646", "confirmed",
    "", "r3_spatial/spatial_dataset_search.csv")
# ---- 6. credibility of the design and of the atlas (round 4)
add("6 credibility (round 4)", "The one-sample-per-time-point pipeline has low realised FDP on dense references with a held-out truth", v("R09"),
    "", "r4_benchmark/benchmark_summary.csv", "r4_benchmark/benchmark_summary.csv; r4_benchmark/benchmark_sensitivity.csv")
add("6 credibility (round 4)", "Replicated bulk cochlear rhythms are carried by within-cell regulation of neural / glial / vascular / bone / strial types, not by composition",
    v("R10"), "", "r4_attribution/*", "r4_attribution/attribution_celltype_enrichment.csv")
add("6 credibility (round 4)", "Replicated bulk BM rhythms reflect both composition and within-cell regulation", v("R11"), "", "r4_attribution/*")
add("6 credibility (round 4)", "Our cochlear labels transfer to independent adult cochlear atlases (marker support, author-label agreement)", v("R15"), "",
    "r4_concordance/*", "r4_concordance/atlas_coverage.csv")
# ---- 7. drivers and conservation (round 4)
add("7 drivers and conservation (round 4)", "Core-clock and heat-shock rhythms are clock outputs (constant-darkness phase test); BM neutrophil programmes are diurnal state programmes",
    v("R16"), "", "r4_drivers/*", "r4_drivers/driver_landscape.csv; r4_drivers/diurnal_circadian_status.csv")
add("7 drivers and conservation (round 4)", "Heat-shock programme follows the internal clock / activity phase in mouse and baboon", v("R12"), "",
    "r4_crossspecies/baboon_heatshock_phase.csv", "r4_crossspecies/baboon_heatshock_phase.csv")
add("7 drivers and conservation (round 4)", "BM gene rhythms are phase-conserved in primate immune tissues", v("R13"), "", "r4_crossspecies/baboon_bm_gene_concordance.csv")
add("7 drivers and conservation (round 4)", "The neutrophil-timer split (cochlea +, marrow −) holds in human blood neutrophils", v("R14"), "",
    "r4_crossspecies/human_neutrophil_timeofday.csv", "r4_crossspecies/human_neutrophil_gene_deltas.csv")
add("7 drivers and conservation (round 4)", "Ageing atlas: rhythmic programmes vs age (descriptive; one library per age)",
    "descriptive" if (RESULTS / "r4_ageing").exists() else "not run", "", "r4_ageing/*")
# ---- 8. B cells (round 5; pre-specified primary hypotheses, decision rule D34)
ph = rd("r5_bcell/primary_hypotheses.csv")
for r in ph.itertuples() if len(ph) else []:
    add("8 B cells (round 5)", f"{r.hypothesis} — {r.tissue}", r.decision,
        f"primary p {r.primary_p:.3f}; BH q {r.q_BH_primary_set:.2f}; agreeing methods {int(r.n_methods_agree)}/{int(r.n_methods)}; window met {bool(r.in_window)}",
        "r5_bcell/primary_hypotheses.csv; r5_bcell/b_multimethod_matrix.csv", "r5_bcell/timeline_24h.csv")
add("8 B cells (round 5)", "Label-free developmental wave: later B neighbourhoods (scVI pseudotime) peak later in the night (BM ZT15 → ZT21)",
    "exploratory", "", "r5_bcell/b_milo_wave.csv", "r5_bcell/timeline_24h.csv")
add("8 B cells (round 5)", "Our B labels vs ImmGen fractions: 'Transitional' ≈ large pre-B (FrC′), 'Mature' ≈ immature (FrE) — naming note, labels unchanged",
    "documented (G41)", "", "r5_bcell/immgen_label_mapping.csv")
add("8 B cells (round 5)", "Recirculating-mature (FrF) B signature peaks at night in independent long-bone BM (GSE198770)", "exploratory (external)", "",
    "r5_bcell/b_external_signature_rhythm.csv")
add("8 B cells (round 5)", "BM B rhythms predict IgA-plasma-cell ZT12 − ZT0 differences, about half as strongly after B-cell Bmal1 deletion (partly B-clock-dependent)",
    "exploratory (external)", "", "r5_bcell/b_external_bmal1ko.csv")
# ---- 9. myeloid cells (round 6; pre-specified primary hypotheses, decision rule D45 / D48)
ph6 = rd("r6_myeloid/m_primary_hypotheses.csv")
for r in ph6.itertuples() if len(ph6) else []:
    add("9 myeloid (round 6)", f"{r.name}", r.decision,
        "" if r.decision == "pending" else f"{r.p_by_tissue}; combined p {r.combined_p:.3f}; BH q {r.q_BH:.2f}; agreeing methods {int(r.n_agreeing_methods_best_tissue)} "
        f"({r.best_tissue}); direction {r.direction_ok}" + (f"; {r.note}" if isinstance(r.note, str) and r.note else ""),
        "r6_myeloid/m_primary_hypotheses.csv; r6_myeloid/m_multimethod_matrix.csv", "r6_myeloid/m_timeline_24h.csv")
ax6 = rd("r6_myeloid/m_axes_rhythm.csv")
if len(ax6):
    x = ax6[(ax6["axis"] == "blood timer ZT13−ZT5 (GSE86619)") & (ax6["group"] == "pooled neutrophils") & (ax6["subset"] == "all cells")].set_index("tissue")
    add("9 myeloid (round 6)", "The blood time-of-day (GSE86619) programme peaks in opposite halves of the day in marrow and cochlear neutrophils",
        "exploratory (M2 partial)", "; ".join(f"{t} ZT{x.loc[t, 'peak_zt']:.1f} (p {x.loc[t, 'p_perm']:.3f})" for t in x.index),
        "r6_myeloid/m_axes_rhythm.csv; r6_myeloid/m_paired_contrast.csv", "r6_myeloid/m_timeline_24h.csv")
tsv6 = rd("r6_myeloid/m_timer_split_verification.csv")
if len(tsv6):
    d_ = tsv6[tsv6["part"].str.startswith("D") & (tsv6["gene_set"] == "rhythmic genes (p < 0.05)")]
    k_ = tsv6[tsv6["part"].str.startswith("A/B") & (tsv6["gene_set"] == "all shared genes") & (tsv6["unit"] == "Neutrophil (Aged)")]
    add("9 myeloid (round 6)", "The neutrophil-timer split (G02: cochlea +, marrow −) is neutrophil-specific, not a sample-level pattern: in each tissue the "
        "key neutrophil states are the most extreme cell types in the expected direction, and in independent long-bone marrow the negative sign "
        "requires neutrophils (lost after neutrophil depletion)", reg["verdict"].get("G02", "not available") if len(reg) else "not available",
        "external BM rhythmic genes: " + "; ".join(f"{r.unit.split(', ')[1]} rho {r.spearman:+.2f}" for r in d_.itertuples()) +
        " | 'Aged' rho (exact time-relabelling p): " + "; ".join(f"{r.tissue} {r.spearman:+.2f} ({r.p_exact_two_sided:.3f})" for r in k_.itertuples()),
        "r6_myeloid/m_timer_split_verification.csv; r2_audit/validation_matrix.csv", "r6_myeloid/m_timer_split_verification.csv")
mw6 = rd("r6_myeloid/m_milo_wave.csv")
if len(mw6):
    x = mw6[mw6["selection"] == "all neutrophil-lineage neighbourhoods"]
    add("9 myeloid (round 6)", "Label-free wave: later neutrophil-lineage neighbourhoods (scVI pseudotime) peak later in the day in both tissues",
        "exploratory", "; ".join(f"{r.tissue} r {r.circ_linear_r:.2f} (p {r.p_perm:.4f})" for r in x.itertuples()), "r6_myeloid/m_milo_wave.csv")
nr6 = rd("r6_myeloid/m_niche_ratio_rhythm.csv")
if len(nr6):
    x = nr6[(nr6["score"] == "cxcr4_cxcr2") & (nr6["subset"] == "all cells") & nr6["tissue"].isin(["bm", "cochlea"])]
    add("9 myeloid (round 6)", "Cochlear neutrophil Cxcr4 / Cxcr2 (retention) peaks in the light phase, antiphase to the blood-timer score (M6 direction not met)",
        "exploratory", "; ".join(f"{r.tissue} ZT{r.peak_zt:.1f} (p {r.p_perm:.3f})" for r in x.itertuples()), "r6_myeloid/m_niche_ratio_rhythm.csv; r6_myeloid/m_primary_m6.csv")
cl6 = rd("r6_myeloid/m_clock_cell_level.csv")
if len(cl6):
    x = cl6[cl6["group"] == "myeloid (pooled)"]
    add("9 myeloid (round 6)", "Marrow myeloid cell clocks track time of day at the cell level, in phase with structural cells; cochlear myeloid clocks are under-powered",
        "exploratory (M7 not supported)", "; ".join(f"{r.tissue} R {r.offset_resultant:.2f} p {r.p_perm:.3f}, offset vs structural {r.offset_minus_structural_h:+.1f} h, "
                                                    f"usable {r.frac_cells_usable:.2f}" for r in x.itertuples()), "r6_myeloid/m_clock_cell_level.csv")
# ---- 12. exploratory reassessment (round 7; 680) — exploratory grades next to the unchanged confirmatory decisions
rg7 = rd("r7_exploratory/ex_claim_regrade.csv")
for _, r in rg7.iterrows() if len(rg7) else []:
    uc = r["unresolved_confounds"]
    add("12 exploratory reassessment (round 7; EXPLORATORY)", f"{r['claim']} — {r['tissue']}",
        f"{r['exploratory_grade']} (confirmatory: {r['confirmatory_decision (unchanged)']})",
        f"primary p {r['primary_p']:.3f}; direction {r['direction']}; supporting lines {int(r['n_supporting_lines'])}; consistency "
        f"{r['directional_consistency (reported, not graded; post hoc)']}" + (f"; unresolved confound: {uc}" if isinstance(uc, str) and uc else ""),
        "r7_exploratory/ex_claim_regrade.csv", "r7_exploratory/ex_claim_regrade.csv")
# ---- 13. Ma 2025 × atlas integration (round 8; 700 / 710 / 720) — EXPLORATORY claim table with timing basis
ct8 = rd("r8_ma_integration/ma_claim_table.csv")
for _, r in ct8.iterrows() if len(ct8) else []:
    add("13 Ma 2025 integration (round 8; EXPLORATORY)", r["claim"], f"{r['updated_exploratory_grade']} (round 7: {r['round7_grade']})",
        f"Ma lines: {int(r['ma_lines_supporting'])} supporting, {int(r['ma_lines_opposing'])} opposing; timing: {r['timing_basis']}"
        + (f"; basis: {r['upgrade_basis']}" if isinstance(r["upgrade_basis"], str) and r["upgrade_basis"] else ""),
        "r8_ma_integration/ma_claim_table.csv; r8_ma_integration/ma_claim_lines.csv; r8_ma_timing/ma_timing_resolution.csv",
        "r8_ma_integration/ma_claim_table.csv")
cnc = pd.DataFrame(rows)
write_csv(cnc, ".", "r3_conclusions.csv", "Concluding synthesis (rounds 1–3): claims by user question with verdicts, key numbers, "
          "evidence and figure-ready tables", required=["section", "claim", "verdict", "evidence_files"])

# ---- noise-response time course
tc = []
g107 = rd("r2_pilots/pilot_noise_geneset_overlap.csv")
if len(g107):
    for r in g107[g107["contrast"].isin(["noise_day_shamsurg", "noise_night_shamsurg"]) & g107["gene_set"].isin(["IEG", "heat shock", "NF-kB", "IL-1"])].itertuples():
        tc.append({"dataset": "GSE107086 (bulk cochlea)", "time_after_noise": "2 h", "exposure_time": "day ZT3" if "day" in r.contrast else "night ZT15",
                   "readout": f"{r.gene_set} genes mean log2FC", "value": r.mean_logFC_in_set, "unit": "log2FC vs sham noise"})
kin = rd("r2_pilots/pilot_noise_kinetics_genesets.csv")
for r in kin[kin["gene_set"].isin(["IEG", "heat shock"])].itertuples() if len(kin) else []:
    tc.append({"dataset": "GSE312476 (bulk cochlea)", "time_after_noise": r.contrast.replace("PN", "").replace("_vs_mock", ""), "exposure_time": "not stated",
               "readout": f"{r.gene_set} genes mean log2FC", "value": r.mean_logFC, "unit": "log2FC vs mock"})
for key, cons, lab_t in [("gse167078", ["noise_3d_vs_sham", "noise_7d_vs_sham", "noise_14d_vs_sham"], "night (from 20:00)"),
                         ("gse346691", ["WT_noise_vs_WT_control"], "not stated"),
                         ("ma2025", ["day_noise_vs_control", "night_noise_vs_control"], "day ZT2–4 / night ZT14–16")]:
    c = rd(f"r3_{key}/composition_contrasts.csv")
    if not len(c):
        continue
    for con in cons:
        x = c[(c["subset"] == "all cells") & (c["contrast"] == con) &
              c["label"].isin(["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Macrophage", "Monocyte (Classical)"])]
        for r in x.itertuples():
            tpoint = {"gse167078": con.split("_vs_")[0].replace("noise_", "").replace("d", " d"), "gse346691": "3 d (series design)",
                      "ma2025": "3 d"}[key]
            tc.append({"dataset": f"{key} (scRNA, our labels)", "time_after_noise": tpoint,
                       "exposure_time": lab_t if key != "ma2025" else ("day" if con.startswith("day") else "night"),
                       "readout": f"{r.label} share log2 ratio", "value": r.log2_ratio, "unit": "log2 ratio vs control"})
write_csv(pd.DataFrame(tc), "r3_response", "noise_response_timecourse.csv",
          "Noise-response summaries across public cochlear datasets ordered by time after exposure (figure-ready)",
          required=["dataset", "time_after_noise", "readout", "value"])
log("\n" + cnc[["section", "claim", "verdict"]].to_string(max_colwidth=90))
