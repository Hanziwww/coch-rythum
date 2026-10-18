"""Final catalogue of every deliverable CSV under results/ (path, module, rows, columns, md5, size, producing script,
description). Descriptions come from the Python write log (cache/manifest_entries.jsonl) or, for R outputs, from the
map below. Output: results/results_manifest.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import MANIFEST, RESULTS, ROOT, log  # noqa: E402

R_DESC = {
    "31_rhythm_genes/rhythm_limma": ("31_rhythm_limma.R", "Primary limma-voom cosinor fits per gene (amplitude, phase, bootstrap CIs, permutation and LOTO statistics, edgeR-QL)"),
    "31_rhythm_genes/permutation_null_counts": ("31_rhythm_limma.R", "Hit counts under each of the 60 dihedral relabeling classes (class 1 = observed)"),
    "31_rhythm_genes/rhythm_global_test": ("31_rhythm_limma.R", "Per-cell-type global permutation test and empirical FDR of rhythmic-gene counts"),
    "32_rhythm_alt/rhythm_alt": ("32_rhythm_alt_methods.R", "Independent rhythm methods per gene: RAIN, JTK_CYCLE, LimoRhyde2 posterior amplitude/phase (LR column empty; see rhythm_lr_*)"),
    "32_rhythm_alt/rhythm_lr": ("32b_lr_rhythmicity.R", "LR rhythmicity test with finite-sample correction (diffCircadian) per gene × cell type"),
    "32_rhythm_alt/lr_tier_impact": ("32b_lr_rhythmicity.R", "Tier-1 calls with the LR test counted as an additional independent method"),
    "33_rhythm_tiers/rhythm_pipeline_simulation": ("33b_simulation.R", "Simulation of the primary rhythm pipeline on the data's own noise: power, FDP, amplitude/phase recovery"),
    "34_clock_network/clock_ccd": ("34_clock_network.R", "Clock correlation distance per tissue × cell type with permutation p"),
    "34_clock_network/clock_network_by_celltype": ("34_clock_network.R", "Clock phase offset, coherence and CCD per tissue × cell type"),
    "35_crosstissue/crosstissue_dryR": ("35_crosstissue_dryR.R", "dryR model selection between cochlea and temporal-BM rhythms per shared cell type and gene"),
    "35_crosstissue/crosstissue_circacompare": ("35_crosstissue_dryR.R", "CircaCompare mesor/amplitude/phase differences for genes with rhythmic dryR models"),
    "35_crosstissue/crosstissue_dryR_model_counts": ("35_crosstissue_dryR.R", "Number of genes per dryR model and cell type"),
    "35_crosstissue/crosstissue_dryR_labelled_counts": ("35b_crosstissue_lr_diff.R", "dryR genes per labelled model class (arrhythmic / cochlea_only / bm_only / shared / different), all and BICW >= 0.5"),
    "35_crosstissue/crosstissue_lr_diff": ("35b_crosstissue_lr_diff.R", "LR_diff (diffCircadian) tests of amplitude and phase differences cochlea vs BM for dryR-rhythmic genes"),
    "36_shared_specific/dream_shared_specific": ("36_shared_specific.R", "dream decomposition of shared vs cell-type-specific rhythm components"),
    "36_shared_specific/mash_posterior": ("36_shared_specific.R", "mashr posterior rhythm effects per gene × cell type"),
    "36_shared_specific/mash_pairwise_sharing": ("36_shared_specific.R", "Pairwise sharing of rhythms between cell types (mashr)"),
    "36_shared_specific/shared_specific_summary": ("36_shared_specific.R", "Summary of shared/specific rhythm counts per tissue"),
    "36_shared_specific/mash_mixture_weights": ("36_shared_specific.R", "mashr mixture weights per covariance component (shared rhythm, cell-type-only, data-driven)"),
    "37_power/power_by_celltype": ("37_power.R", "Analytic power of the moderated cosinor F test per cell type × expression bin"),
    "38_cell_level/cell_level_glmm": ("38_cell_level_glmm.R", "Cell-level NB GLMM (sample random effect) rhythm tests for candidate genes"),
    "39_sensitivity/depth_matched_summary": ("39_sensitivity.R", "Rhythm counts, pi1 and phase concentration per cell type before vs after depth matching (binomial thinning)"),
    "39_sensitivity/depth_matched_genes": ("39_sensitivity.R", "Gene-level cosinor after depth matching (first thinning) next to the primary fit"),
    "39_sensitivity/statemix_sensitivity": ("39_sensitivity.R", "Gene-level rhythm tests with the within-type state score as covariate"),
    "39_sensitivity/statemix_summary": ("39_sensitivity.R", "Share of Tier-1/2 genes surviving the state-mix covariate per cell type"),
    "39_sensitivity/pseudoreplicate_rhythm": ("39_sensitivity.R", "Cell-subsampling pseudo-replicate rhythm tests (anticonservative) for Tier-1 and clock genes"),
    "54_abundance_power/abundance_power": ("54_abundance_power.R", "Power of abundance-rhythm tests (propeller moderated; CLR unmoderated) per cell type"),
    "75_deconvolution/bayesprism_fractions": ("75b_bayesprism.R", "BayesPrism cell-type fractions of external bulk samples (atlas reference)"),
    "75_deconvolution/bayesprism_abundance_rhythm": ("75b_bayesprism.R", "Cosinor rhythm of BayesPrism fractions per condition and cell type"),
    "75_deconvolution/bayesprism_condition_effects": ("75b_bayesprism.R", "Condition effects (mean shift, rhythm difference) on BayesPrism fractions"),
    "44_motifs/motif_enrichment": ("44_motif_enrichment.R", "RcisTarget motif enrichment (mm10 v10 clustered motifs) of rhythmic genes by phase bin and per cell type"),
    "51_abundance_models/abundance_rhythm_propeller": ("51_abundance_models.R", "propeller-style logit-proportion cosinor tests per cell type"),
    "51_abundance_models/abundance_rhythm_sccomp": ("51_abundance_models.R", "sccomp beta-binomial cosinor effects per cell type"),
    "53_milo/milo_nhoods": ("53_milo_rhythm.R", "Milo neighbourhoods with majority gold-standard label"),
    "53_milo/milo_da_results": ("53_milo_rhythm.R", "Neighbourhood-level rhythmic abundance tests with spatial FDR"),
    "67_multinichenet/multinichenet_prioritization": ("67_multinichenet.R", "MultiNicheNet prioritised interactions, rest (ZT3/7/11) vs active (ZT15/19/23); cochlea: immune receivers, <= 800 cells per sample × type"),
    "67_multinichenet/multinichenet_ligand_activity": ("67_multinichenet.R", "NicheNet ligand activities per receiver and contrast"),
    "71_external/external_rhythm": ("71_external_rhythm.R", "Cosinor rhythm statistics of external replicated bulk datasets"),
    "71_external/external_dryR": ("71_external_rhythm.R", "dryR condition comparison in external datasets (sham vs ADX; control vs depletion)"),
    "71_external/external_manifest": ("71_external_rhythm.R", "External datasets used: samples, time points, replicates"),
}
R2_DESC = {   # round-2 R outputs, checked before the generic round-1 keys (first substring match wins)
    "53_milo/milo_counts_": ("53b_milo_robustness.R", "Milo neighbourhood x sample cell counts (primary scANVI neighbourhoods, rebuilt and asserted equal to 53)"),
    "53_milo/milo_robustness_summary": ("53b_milo_robustness.R", "Milo rhythmic neighbourhoods per tissue and majority type vs the exact 60-class time-relabelling null (spatial-FDR calls and LOTO-robust calls)"),
    "53_milo/milo_robustness_": ("53b_milo_robustness.R", "Per neighbourhood: sample dominance, leave-one-time-point-out refits (max p, min amplitude ratio, phase s.d.), single-sample-driven and robust flags"),
    "53_milo/milo_relabel_null": ("53b_milo_robustness.R", "Rhythmic neighbourhood counts per tissue, majority type and dihedral relabelling class (class 1 = observed)"),
    "53_milo/milo_relabel_peaks_": ("53c_milo_wave_relabel.R", "scVI Milo neighbourhood peak ZT and p under each of the 60 dihedral relabellings of the sampling times (class 1 = observed)"),
    "53_milo/milo_audit_symmetry": ("53e_milo_verification.R", "Audit of the relabelling machinery: cosinor F invariant under the 12 rotations / reflections of the sampling times; 60 distinct class designs"),
    "53_milo/milo_relabel_fits_": ("53e_milo_verification.R", "Milo neighbourhood rhythm test (edgeR QL) refitted for each of the 60 dihedral relabelling classes: F, p, peak, spatial FDR"),
    "53_milo/milo_positive_control": ("53e_milo_verification.R", "Positive control of the edgeR relabelling statistics: rhythms injected into 5 % of real neighbourhoods; exact p of the injected class per fold change"),
    "53_milo/milo_membership_": ("53e_milo_verification.R", "Cells of the rhythmic Milo neighbourhoods (spatial FDR < 0.1; primary scANVI neighbourhoods)"),
    "r9_figures/f3b_tissue_timecourses": ("903_figure_rderive.R", "Fig. 3b: per-sample voom log2 CPM of heat-shock, immediate-early, NF-kB and clock genes in the tissue pseudobulk with the stored cosinor fit (refit asserted equal)"),
    "r9_figures/f4g_neutrophil_state_genes": ("903_figure_rderive.R", "Fig. 5e: per-sample voom log2 CPM of the Tier-1 genes of aged and activated marrow neutrophils with the stored cosinor fit (refit asserted equal)"),
    "53_milo/milo_nhoods_bm_scvi": ("53_milo_rhythm.R (scVI latent)", "Milo neighbourhoods on the unsupervised scVI embedding (round-2 sensitivity)"),
    "53_milo/milo_nhoods_cochlea_scvi": ("53_milo_rhythm.R (scVI latent)", "Milo neighbourhoods on the unsupervised scVI embedding (round-2 sensitivity)"),
    "53_milo/milo_da_results_bm_scvi": ("53_milo_rhythm.R (scVI latent)", "Milo rhythmic neighbourhood tests on the unsupervised scVI embedding (round-2 sensitivity)"),
    "53_milo/milo_da_results_cochlea_scvi": ("53_milo_rhythm.R (scVI latent)", "Milo rhythmic neighbourhood tests on the unsupervised scVI embedding (round-2 sensitivity)"),
    "r2_figure/abundance_multimodel_long": ("202_abundance_multimodel.R", "Abundance rhythm of every cell type under 10 models / reference frames (p, peak ZT and clock)"),
    "r2_figure/abundance_multimodel_summary": ("202_abundance_multimodel.R", "Per series: core-model support, peak concordance, permutation / LOTO / bootstrap and verdict"),
    "r2_cochlea/cochlea_relaxed_tier": ("211_cochlea_relaxed_tier.R", "Cochlear gene × cell-type rhythm statistics with supports, rule ladder R0–R4, calibrated Tier C flag and annotations"),
    "r2_cochlea/cochlea_relaxed_calibration": ("211_cochlea_relaxed_tier.R", "Observed vs null (59 relabelings) calls per cell type × rule; empirical FDR with and without tissue-pattern-like genes"),
    "r2_cochlea/cochlea_relaxed_rule_selection": ("211_cochlea_relaxed_tier.R", "Rule ladder: definitions, calls, empirical FDR and the selected (calibrated) Tier C rule"),
    "r2_cochlea/cochlea_relaxed_simulation": ("211_cochlea_relaxed_tier.R", "Injected-rhythm simulation: power and FDP of the internal Tier C criteria"),
    "r2_cochlea/motif_enrichment_tierC": ("44_motif_enrichment.R (TIERC=1)", "RcisTarget motif enrichment of Tier C genes by phase bin and cell type (exploratory)"),
    "r2_external/external_rhythm_": ("221_external_rhythm.R", "Cosinor (24 h; 12-h harmonic where replicated) per gene in round-2 external datasets"),
    "r2_external/external_dryR_": ("221_external_rhythm.R", "dryR condition comparison (control vs depletion / KO) in round-2 external datasets"),
    "r2_external/external_manifest_r2": ("221_external_rhythm.R", "Round-2 external datasets: samples, time points, replicates, time type"),
    "r2_phase/phase_ci_coverage": ("232_phase_ci_coverage.R", "Coverage of the delta-method 95 % phase CI for injected rhythms on the data's own noise"),
    "r2_audit/dryr_null_check": ("243_dryr_null_check.R", "dryR model-class counts under null relabelling of cochlear time vs observed (3 cell types × 3 classes)"),
    "r2_pilots/pilot_GSE107086_de": ("251_pilot_noise_bulk.R", "GSE107086 limma-voom contrasts: noise by time of day and surgery, interactions, night − day baseline"),
    "r2_pilots/pilot_GSE312476_de": ("251_pilot_noise_bulk.R", "GSE312476 limma-trend: 0 / 4 / 8 h after noise vs mock"),
    "r2_pilots/cochlea_tissue_pseudobulk_cosinor": ("251_pilot_noise_bulk.R", "Tissue-level pseudobulk cosinor of our cochlea (all cell types summed) for external predictions"),
}
R3_DESC = {
    "r3_bm/bm_relaxed_tier": ("211_cochlea_relaxed_tier.R (TISSUE=bm)", "BM gene × cell-type rhythm statistics with supports (GSE198770 replicates 1–2, cochlea), rule ladder, selected tier"),
    "r3_bm/bm_relaxed_calibration": ("211_cochlea_relaxed_tier.R (TISSUE=bm)", "BM rule ladder: observed vs null (59 relabelings) calls per cell type × rule; empirical FDR"),
    "r3_bm/bm_relaxed_rule_selection": ("211_cochlea_relaxed_tier.R (TISSUE=bm)", "BM rule ladder definitions, calls, FDR and the selected rule"),
    "r3_bm/bm_relaxed_simulation": ("211_cochlea_relaxed_tier.R (TISSUE=bm)", "Injected-rhythm simulation of the BM relaxed criteria: power and FDP"),
    "r3_bm/bm_tissue_pseudobulk_cosinor": ("211_cochlea_relaxed_tier.R (TISSUE=bm)", "Tissue-level pseudobulk cosinor of our BM (all cell types summed)"),
    "/pseudobulk_de": ("311_external_pseudobulk_de.R", "External single-cell data: pseudobulk limma-voom DE per transferred label and contrast"),
    "r5_bcell/b_nichenet_ligand_activity": ("541_nichenet_b.R", "NicheNet ligand activity for the night- / day-peaking programme of each B stage and cochlear partner (AUPR-corrected, rank, percentile)"),
    "r6_myeloid/m_nichenet_ligand_activity": ("651_nichenet_myeloid.R", "NicheNet ligand activity for the night- / day-peaking programme of each myeloid receiver (AUPR-corrected, rank, percentile; M6 CXCL12 percentile)"),
}
R_DESC = {**R3_DESC, **R2_DESC, **R_DESC}
logged = {}
if MANIFEST.exists():
    for line in open(MANIFEST):
        e = json.loads(line)
        logged[e["path"]] = e                      # last write wins


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for p in sorted(RESULTS.rglob("*.csv")):
    rel = str(p.relative_to(ROOT))
    if p.name == "results_manifest.csv":
        continue
    with open(p) as fh:
        header = fh.readline().rstrip("\n").split(",")
        n = sum(1 for _ in fh)
    e = logged.get(rel)
    script, desc = (e["script"], e["description"]) if e else next(
        ((s, d) for k, (s, d) in R_DESC.items() if k in rel), ("", ""))
    rows.append({"path": rel, "module": p.relative_to(RESULTS).parts[0], "rows": n, "n_columns": len(header),
                 "columns": ";".join(header[:60]) + (";..." if len(header) > 60 else ""), "size_mb": round(p.stat().st_size / 1e6, 3),
                 "md5": md5(p), "script": script, "description": desc})
man = pd.DataFrame(rows)
man.to_csv(RESULTS / "results_manifest.csv", index=False)
log(f"{len(man)} CSV files catalogued; {man['size_mb'].sum():.0f} MB; undocumented: {(man['description'] == '').sum()}")
