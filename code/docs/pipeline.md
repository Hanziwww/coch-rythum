# Pipeline: modules, scripts and outputs

Generated from `results/results_manifest.csv` (every registered table with its producing script) and `results/r9_figures/panel_index.csv` (figure panels and their tables). Scripts are in `code/scripts/`; run them from the project root (see the README). Purposes come from the script headers.

## Script groups

| Group | Scripts | First | Last |
| --- | --- | --- | --- |
| setup and data audit | 7 | 00_inspect_obs.py | 13_clock_controls.py |
| dimension reduction and embeddings | 6 | 20_dr_stageA.py | 23_dr_finalize.py |
| pseudobulk and gene rhythms | 14 | 30_pseudobulk.py | 39_sensitivity.R |
| pathways, enrichment and motifs | 8 | 40_download_resources.sh | 44_promoter_motifs.py |
| composition and neighbourhoods | 11 | 50_abundance_frames.py | 54_abundance_power.R |
| migration axes and communication | 10 | 60_migration_models.py | 69_migration_methods_review.py |
| external circadian data | 10 | 70_external_prepare.py | 77_neutrophil_timer.py |
| resource and applied modules | 5 | 80_atlas_catalogue.py | 84_gwas_overlap.py |
| bookkeeping (manifest, key findings, plan status) | 3 | 97_plan_status.py | 99_results_manifest.py |
| validation (time axis, external validation, audit, evidence register) | 22 | 200_time_axis_audit.py | 252_pilot_noise_linking.py |
| external single-cell, spatial and noise data | 19 | 300_external_sc_ingest.py | 360_conclusions.py |
| design benchmark, attribution, drivers, cross-species, atlas concordance | 9 | 410_design_benchmark.py | 460_resource_tables.py |
| B-lineage hypotheses | 9 | 510_b_composition.py | 570_b_synthesis.py |
| myeloid hypotheses and neutrophil timer | 13 | 610_myeloid_refs.py | 680_exploratory_reassessment.py |
| independent noise cohort (exploratory) | 3 | 700_ma_timing.py | 720_ma_claim_synthesis.py |
| figure data, figure QC, manuscript, tables and release | 10 | 900_figure_data.py | 909_source_data.py |
| main figures | 7 | 910_fig1_atlas.py | 917_sfig1_flow_gating.py |
| Extended Data figures | 10 | 920_ed1_qc.py | 929_ed10_ma2025.py |
| imaging and flow cytometry | 9 | 930_image_panels.py | 939_image_robustness.py |
| image verification, robustness, 3D image inventory and Supplementary imaging figures | 12 | 940_flow_gate_robustness.py | 959_imaging_io.py |
| parabiosis cell counts with the upgraded rules | 4 | 960_survey_cells.py | 963_survey_cells_extra.py |
| Science version: figures, Supplementary Materials, manuscript build and package | 6 | 970_science_figures.py | 979_science_package.py |

## Scripts and their outputs

### `00_inspect_obs.py`

Inspect the cochlear atlas: obs columns, sample-level metadata, and candidate circadian/time-of-day labels.

### `01_inspect_daytime.py`

Cross-tabulate the circadian label (day_time) against sample-level metadata, and check whether .X holds raw counts.

### `01_software_versions.py`

Module 0.1 — software versions and compute environment used for all analyses.

Outputs (2) in `results/00_env/`: `compute_environment.csv`, `software_versions.csv`.

### `02_subset_circadian.py`

Step 1: keep only the circadian time-course samples (cells with a day_time label).

### `10_audit.py`

Module 0.2 — data audit of the circadian subset (read-only on the data; CSV outputs only).

Outputs (6) in `results/00_audit/`: `celltype_counts_by_sample.csv`, `gene_filter_audit.csv`, `label_markers.csv`, `label_support_qc.csv`, `label_support_summary.csv`, `sample_qc_summary.csv`.

### `12_technical_audit.py`

Module 0.3 — technical negative-control audit (reported only, never used as model covariates).

Outputs (2) in `results/01_technical_audit/`: `technical_audit.csv`, `technical_audit_rhythm.csv`.

### `13_clock_controls.py`

Module 0.4a — core-clock positive controls and time-label checks (data-internal part).

Outputs (4) in `results/02_clock_controls/`: `clock_crosstissue_concordance.csv`, `clock_gene_profiles.csv`, `clock_gene_rhythm.csv`, `clock_phase_order.csv`.

### `20_dr_stageA.py`

Module 1, Stage A — feature selection / normalisation screen with PCA only.

Outputs (1) in `results/20_dr/`: `dr_hvg_sets.csv`.

### `20b_select_features.py`

Module 1, Stage A selection — choose the feature/normalisation configuration per family.

Outputs (1) in `results/20_dr/`: `stageA_selection.csv`.

### `20c_select_stageC.py`

Module 1 helper — pick Stage C candidates from the benchmark metrics (pre-registered composite and time-loss rule).

### `21_dr_train.py`

Module 1, Stages B/C — train integration/embedding methods on the Stage-A feature set of a family.

### `22_dr_metrics.py`

Module 1 — evaluation of every cached embedding of a family on its fixed label-stratified subsample.

Outputs (3) in `results/20_dr/`: `dr_benchmark_metrics_bm.csv`, `dr_benchmark_metrics_cochlea.csv`, `dr_benchmark_metrics_joint.csv`.

### `23_dr_finalize.py`

Module 1 — final embeddings: pre-registered composite selection, then neighbours/UMAP on all cells.

Outputs (8) in `results/20_dr/`: `dr_benchmark_summary.csv`, `embedding_bm_latent.csv`, `embedding_cochlea_latent.csv`, `embedding_joint_latent.csv`, `neighbors_summary.csv`, `umap_bm.csv`, `umap_cochlea.csv`, `umap_joint.csv`.

### `30_pseudobulk.py`

Module 2.1 — pseudobulk count matrices per tissue × gold-standard group × sample.

Outputs (80) in `results/30_pseudobulk/`: `bm__cell_class__Cycling.csv`, `bm__cell_class__Epithelial.csv`, `bm__cell_class__Erythroid.csv`, `bm__cell_class__Lymphoid.csv`, `bm__cell_class__Myeloid.csv`, `bm__cell_class__Progenitor.csv`, `bm__cell_class__Stromal.csv`, `bm__cell_type__B_Cell_Mature.csv`, `bm__cell_type__B_Cell_Na_ve.csv`, `bm__cell_type__B_Cell_Transitional.csv`, `bm__cell_type__Cycling_Cell.csv`, `bm__cell_type__Epithelial_Cell_Non_sensory.csv` and 68 more.

### `31_rhythm_limma.R`

Module 2.2/2.4 — primary gene-rhythm model per tissue × group (limma-voom cosinor, LimoRhyde framework).

Outputs (8) in `results/31_rhythm_genes/`: `permutation_null_counts_cell_class.csv`, `permutation_null_counts_cell_type.csv`, `rhythm_global_test_cell_class.csv`, `rhythm_global_test_cell_type.csv`, `rhythm_limma_cell_class_bm.csv`, `rhythm_limma_cell_class_cochlea.csv`, `rhythm_limma_cell_type_bm.csv`, `rhythm_limma_cell_type_cochlea.csv`.

### `32_rhythm_alt_methods.R`

Module 2.3 — independent rhythm methods on the same pseudobulk data as 31_rhythm_limma.R.

Outputs (4) in `results/32_rhythm_alt/`: `rhythm_alt_cell_class_bm.csv`, `rhythm_alt_cell_class_cochlea.csv`, `rhythm_alt_cell_type_bm.csv`, `rhythm_alt_cell_type_cochlea.csv`.

### `32b_lr_rhythmicity.R`

Module 2.3 (supplement) — likelihood-ratio rhythmicity test with finite-sample correction (diffCircadian LR_rhythmicity, Ding 2021) on TMM log2-CPM of each pseudobulk series; it was not installed when 32/33 ran, so the tier definitions (33) use edgeR-QL, RAIN and JTK as independent methods.

Outputs (3) in `results/32_rhythm_alt/`: `lr_tier_impact.csv`, `rhythm_lr_cell_type_bm.csv`, `rhythm_lr_cell_type_cochlea.csv`.

### `33_rhythm_tiers.py`

Module 2.5 — merge primary and independent rhythm results, assign evidence tiers, summarise per cell type.

Outputs (6) in `results/33_rhythm_tiers/`: `rhythm_genes_all_cell_class_bm.csv`, `rhythm_genes_all_cell_class_cochlea.csv`, `rhythm_genes_all_cell_type_bm.csv`, `rhythm_genes_all_cell_type_cochlea.csv`, `rhythm_summary_by_celltype_cell_class.csv`, `rhythm_summary_by_celltype_cell_type.csv`.

### `33b_simulation.R`

Verification — simulation of the rhythm pipeline on the dataset's own noise.

Outputs (1) in `results/33_rhythm_tiers/`: `rhythm_pipeline_simulation.csv`.

### `34_clock_network.R`

Module 2.8 — clock network per tissue × cell type.

Outputs (2) in `results/34_clock_network/`: `clock_ccd.csv`, `clock_network_by_celltype.csv`.

### `35_crosstissue_dryR.R`

Module 2.6 — differential rhythmicity between the independent cochlea and temporal-BM cohorts for cell types included (status 'primary') in both tissues.

Outputs (4) in `results/35_crosstissue/`: `crosstissue_circacompare.csv`, `crosstissue_dryR.csv`, `crosstissue_dryR_labelled_counts.csv`, `crosstissue_dryR_model_counts.csv`.

### `35b_crosstissue_lr_diff.R`

Module 2.6 (supplement) — labelled dryR model classes and likelihood-ratio tests of differential amplitude and phase (diffCircadian::LR_diff, finite-sample corrected; Ding 2021) between the cochlea and temporal-BM cohorts. dryR model numbers are mapped to labels from which tissue's rhythm parameters

Outputs (1) in `results/35_crosstissue/`: `crosstissue_lr_diff.csv`.

### `36_shared_specific.R`

Module 2.7 — shared vs cell-type-specific rhythm components across cell types within each tissue.

Outputs (9) in `results/36_shared_specific/`: `dream_shared_specific_bm.csv`, `dream_shared_specific_cochlea.csv`, `mash_mixture_weights_bm.csv`, `mash_mixture_weights_cochlea.csv`, `mash_pairwise_sharing_bm.csv`, `mash_pairwise_sharing_cochlea.csv`, `mash_posterior_bm.csv`, `mash_posterior_cochlea.csv`, `shared_specific_summary.csv`.

### `37_power.R`

Module 2.10 — power of the primary moderated cosinor F test per tissue × cell type × expression bin.

Outputs (1) in `results/37_power/`: `power_by_celltype.csv`.

### `38_cell_level_glmm.R`

Module 2.3 (sensitivity) — cell-level negative-binomial GLMM with a sample random intercept (NEBULA-type model; NEBULA could not be installed, glmmTMB is used instead) for Tier-1/2 candidates and core-clock genes. count_ic ~ cos(ωt) + sin(ωt) + offset(log library size) + (1 | sample), family nbinom2

Outputs (1) in `results/38_cell_level/`: `cell_level_glmm.csv`.

### `38_export_cells.py`

Export per-cell counts of candidate genes (Tier-1/2 rhythmic genes + core clock) for the cell-level GLMM (38_cell_level_glmm.R): <= 300 random cells per cell type × sample (seed 1), raw counts, library size, sample, ZT.

### `39_sensitivity.R`

Module 2.3/2.4 — sensitivity analyses of the primary pseudobulk rhythm calls (31/33).

Outputs (6) in `results/39_sensitivity/`: `depth_matched_genes_bm.csv`, `depth_matched_genes_cochlea.csv`, `depth_matched_summary.csv`, `pseudoreplicate_rhythm.csv`, `statemix_sensitivity.csv`, `statemix_summary.csv`.

### `40_download_resources.sh`

Download prior-knowledge resources (Module 3), external circadian datasets (Module 5.4) and validation data.

### `41_convert_progeny.R`

Convert the PROGENy mouse model (.rda from the saezlab/progeny GitHub repository) to CSV for 41_pathway_activity.py.

### `41_pathway_activity.py`

Module 3a — pathway and TF-activity rhythms per tissue × cell type.

Outputs (8) in `results/41_pathway/`: `cell_score_rhythm.csv`, `cell_score_sample_medians.csv`, `pathway_rhythm.csv`, `pathway_rhythm_depth_matched.csv`, `pathway_scores.csv`, `pathway_scores_depth_matched.csv`, `resource_summary.csv`, `resource_summary_depth_matched.csv`.

### `42_rhythm_enrichment.py`

Module 3b/3c — competitive enrichment and phase-set enrichment (PSEA) of rhythmic genes per tissue × cell type.

Outputs (4) in `results/42_enrichment/`, `results/r2_cochlea/`: `psea_phase_enrichment.csv`, `rhythmic_gene_enrichment.csv`, `psea_phase_enrichment_tierC.csv`, `rhythmic_gene_enrichment_tierC.csv`.

### `42b_ranked_enrichment.py`

Module 3b (threshold-free) — competitive, expression-stratified enrichment of rhythmicity in gene sets.

Outputs (1) in `results/42_enrichment/`: `rhythmicity_ranked_enrichment.csv`.

### `43_pathway_phase.py`

Module 3c/3d — cell-type-specific (within-sample-centred) pathway rhythms, pathway phase map, TF activity–mRNA lag.

Outputs (7) in `results/43_pathway_phase/`: `pathway_phase_consistency.csv`, `pathway_phase_map.csv`, `pathway_rhythm_centred.csv`, `pathway_rhythm_centred_depth_matched.csv`, `pathway_robust_counts.csv`, `tf_activity_mrna_lag.csv`, `tf_lag_summary.csv`.

### `44_motif_enrichment.R`

Module 2.9/3d — cis-regulatory motif enrichment (RcisTarget, mm10 500 bp up / 100 bp down promoter rankings, motif collection v10 clustered) of rhythmic genes grouped by peak phase, per tissue.

Outputs (1) in `results/44_motifs/`: `motif_enrichment.csv`.

### `44_promoter_motifs.py`

Module 2.9 / 3d — promoter-motif driver attribution of rhythmic genes (replaces the RcisTarget step: the 0.5-GB cisTarget ranking database could not be downloaded at the available bandwidth).

Outputs (5) in `results/44_motifs/`: `promoter_motif_enrichment.csv`, `promoter_motif_enrichment_tissue.csv`, `promoter_motif_phase.csv`, `promoter_motif_phase_bins.csv`, `promoter_motif_presence.csv`.

### `50_abundance_frames.py`

Module 4 — abundance and distribution rhythms in several compositional reference frames.

Outputs (3) in `results/50_abundance/`: `abundance_rhythm_frames.csv`, `composition_frames.csv`, `reference_frame_stability.csv`.

### `51_abundance_models.R`

Module 4 — count-based compositional rhythm models per tissue.

Outputs (2) in `results/51_abundance_models/`: `abundance_rhythm_propeller.csv`, `abundance_rhythm_sccomp.csv`.

### `52_marrow_decomposition.py`

Module 4 core — marrow-vs-tissue decomposition of immune abundance in the cochlear system (Bayesian, numpyro).

Outputs (3) in `results/52_marrow_decomposition/`: `marrow_content_rhythm.csv`, `mixture_model_fits.csv`, `tissue_residual_rhythm.csv`.

### `53_milo_rhythm.R`

Module 4 — neighbourhood-level rhythmic abundance (Milo; Dann 2022) on the selected per-tissue embeddings.

Outputs (4) in `results/53_milo/`: `milo_da_results_bm.csv`, `milo_da_results_cochlea.csv`, `milo_nhoods_bm.csv`, `milo_nhoods_cochlea.csv`.

### `53b_milo_robustness.R`

Module 4 / robustness of the Milo neighbourhood rhythms (G08).

Outputs (6) in `results/53_milo/`: `milo_counts_bm.csv`, `milo_counts_cochlea.csv`, `milo_relabel_null.csv`, `milo_robustness_bm.csv`, `milo_robustness_cochlea.csv`, `milo_robustness_summary.csv`.

### `53c_milo_wave_relabel.R`

Exact time-relabelling null for the Milo production waves (Fig. 5d; M1 / M3 / B02 method lines).

Outputs (2) in `results/53_milo/`: `milo_relabel_peaks_bm_scvi.csv`, `milo_relabel_peaks_cochlea_scvi.csv`.

### `53d_milo_wave_null.py`

Exact time-relabelling null of the Milo production waves (Fig. 5d; M1 / M3 / B02 method lines).

Outputs (2) in `results/53_milo/`: `milo_wave_null_summary.csv`, `milo_wave_relabel_null.csv`.

### `53e_milo_verification.R`

Verification of the Milo exact-null result.

Outputs (8) in `results/53_milo/`: `milo_audit_symmetry.csv`, `milo_membership_bm.csv`, `milo_membership_cochlea.csv`, `milo_positive_control.csv`, `milo_relabel_fits_bm_scanvi.csv`, `milo_relabel_fits_bm_scvi.csv`, `milo_relabel_fits_cochlea_scanvi.csv`, `milo_relabel_fits_cochlea_scvi.csv`.

### `53f_milo_verification.py`

Verification of the Milo exact-null result, Python part (after 53e_milo_verification.R).

Outputs (4) in `results/53_milo/`: `milo_verification_dominance.csv`, `milo_verification_exact.csv`, `milo_verification_external.csv`, `milo_verification_summary.csv`.

### `53g_milo_exact_r2.py`

Dispersion-free exact relabelling test of Milo neighbourhood rhythms, with a positive control.

Outputs (3) in `results/53_milo/`: `milo_r2_exact.csv`, `milo_r2_positive_control.csv`, `milo_r2_variants.csv`.

### `54_abundance_power.R`

Module 4 — power of abundance-rhythm tests per tissue × cell type (cf. 37_power.R for genes).

Outputs (1) in `results/54_abundance_power/`: `abundance_power.csv`.

### `60_migration_models.py`

Module 5.1 — abundance-based migration modelling between temporal-bone marrow (BM) and the cochlear system.

Outputs (3) in `results/60_migration/`: `compartment_model_compare.csv`, `compartment_model_fits.csv`, `phase_relationships.csv`.

### `61_lineage_flux.py`

Module 5.1 (lineage flux) — time-gated production, maturation, release and return along temporal-BM lineages.

Outputs (3) in `results/61_lineage_flux/`: `lineage_flux_fits.csv`, `lineage_flux_grid.csv`, `lineage_flux_nonlinear.csv`.

### `63_ccc_liana.py`

Module 5.2a — per-sample cell–cell communication with LIANA+ (consensus of CellPhoneDB-, CellChat-, NATMI-, Connectome-, SingleCellSignalR-type scores; mouse consensus resource), per tissue, by sample (= ZT).

Outputs (2) in `results/63_ccc/`: `ccc_scores_by_sample_bm.csv`, `ccc_scores_by_sample_cochlea.csv`.

### `64_ccc_rhythm.py`

Module 5.2b — rhythmic communication.

Outputs (5) in `results/64_ccc_rhythm/`: `ccc_rhythm_bm.csv`, `ccc_rhythm_cochlea.csv`, `tensor_factor_rhythm.csv`, `tensor_factors_bm.csv`, `tensor_factors_cochlea.csv`.

### `65_migration_axes.py`

Module 5.2c — curated homing / recruitment / retention / niche axes across the cochlea–BM system.

Outputs (2) in `results/65_migration_axes/`: `axes_vs_abundance_phase.csv`, `migration_axes_rhythm.csv`.

### `65b_axes_category_summary.py`

Module 5.2c (summary) — category-level evidence for rhythmic migration/niche axes.

Outputs (1) in `results/65_migration_axes/`: `migration_axes_category_summary.csv`.

### `66_state_continua.py`

Module 5.2d — receiver-side migratory competence and arrival-state continua (per cell), plus BM-likeness.

Outputs (7) in `results/66_state_continua/`: `bm_likeness_by_cell.csv`, `bm_likeness_by_sample.csv`, `bm_likeness_rhythm.csv`, `state_continuum_shifts.csv`, `state_score_rhythm.csv`, `state_scores_by_cell.csv`, `state_scores_by_sample.csv`.

### `67_export_sce.py`

Export raw counts + cell metadata (MatrixMarket) for 67_multinichenet.R (internal input, not a deliverable).

### `67_multinichenet.R`

Module 5.2 — MultiNicheNet: prioritised ligand → receptor → target links for rest (ZT3/7/11) vs active (ZT15/19/23) phase contrasts per tissue (3 vs 3 pooled samples; pseudobulk edgeR DE via muscat; NicheNet v2 mouse prior).

Outputs (4) in `results/67_multinichenet/`: `multinichenet_ligand_activity_bm.csv`, `multinichenet_ligand_activity_cochlea.csv`, `multinichenet_prioritization_bm.csv`, `multinichenet_prioritization_cochlea.csv`.

### `69_migration_methods_review.py`

Module 5.3 — review of methods for studying immune-cell exchange between the cochlea and the adjacent temporal-bone marrow: what each method tests, what it needs, whether it is feasible with this dataset (processed counts only, one pooled sample per ZT and tissue, cochlea and BM from different perfu

Outputs (1) in `results/69_methods_review/`: `migration_methods_review.csv`.

### `70_external_prepare.py`

Module 5.4 — prepare external circadian datasets into a common format.

Outputs (1) in `results/71_external/`: `external_prepare_report.csv`.

### `71_external_rhythm.R`

Module 5.4a — harmonised rhythm analysis of external circadian bulk datasets (with biological replicates).

Outputs (5) in `results/71_external/`: `external_dryR_GSE198770_bone_marrow.csv`, `external_dryR_GSE88954_cochlea.csv`, `external_manifest.csv`, `external_rhythm_GSE198770.csv`, `external_rhythm_GSE88954.csv`.

### `72_phasor_decomposition.py`

Module 5.4d — phasor decomposition of tissue-level (bulk-like) rhythms into cell-type expression vs abundance terms.

Outputs (5) in `results/72_phasor/`: `phasor_celltype_contributions_bm.csv`, `phasor_celltype_contributions_cochlea.csv`, `phasor_decomposition_bm.csv`, `phasor_decomposition_cochlea.csv`, `phasor_validation.csv`.

### `73_meta_cosinor.py`

Module 5.4e — meta-cosinor: combine our tissue-level rhythms with external replicated bulk data.

Outputs (3) in `results/73_meta/`: `meta_cosinor_bm.csv`, `meta_cosinor_cochlea.csv`, `meta_cosinor_summary.csv`.

### `74_timetelling.py`

Module 0.4b / 5.4g — molecular time-telling of our samples from external circadian references.

Outputs (3) in `results/74_timetelling/`: `reference_validation.csv`, `sample_time_prediction.csv`, `timetelling_summary.csv`.

### `75_deconvolution.py`

Module 5.4c — deconvolution of external circadian bulk data with our atlas as reference (NNLS; complements InstaPrism in 75_deconvolution_instaprism.R when that package is available).

Outputs (4) in `results/75_deconvolution/`: `deconvolution_abundance_rhythm.csv`, `deconvolution_condition_effects.csv`, `deconvolution_fractions.csv`, `deconvolution_reference_markers.csv`.

### `75b_bayesprism.R`

Module 5.4c — BayesPrism deconvolution of external circadian bulk data with our atlas as reference (BM reference for GSE198770; cochlea reference for GSE88954 when prepared).

Outputs (3) in `results/75_deconvolution/`: `bayesprism_abundance_rhythm.csv`, `bayesprism_condition_effects.csv`, `bayesprism_fractions.csv`.

### `75b_export_reference.py`

Export a single-cell reference subsample for BayesPrism (<= 400 cells per cell type per tissue, seed 1; raw counts, protein-coding-like genes only: Gm*/Rik/mt-/ribosomal genes removed).

### `76_dependence_mapping.py`

Module 5.4f — glucocorticoid (GC)- and neutrophil-dependence of rhythmic genes, mapped onto our cell types.

Outputs (4) in `results/76_dependence/`: `dependence_gene_classes.csv`, `dependence_phase_concordance.csv`, `gc_dependence_mapping.csv`, `neutrophil_dependence_mapping.csv`.

### `77_neutrophil_timer.py`

Module 5.4 (E3) — the neutrophil timer: GSE86619 (Adrover 2019) blood neutrophils, WT at ZT5 and ZT13 (n = 3 each) and neutrophil-specific Bmal1 (Arntl), Cxcr2 and Cxcr4 deletion (Mrp8-Cre) at ZT5, versus our neutrophil rhythms.

Outputs (3) in `results/77_neutrophil_timer/`: `neutrophil_timer_concordance.csv`, `neutrophil_timer_gene_table.csv`, `neutrophil_timer_ko_enrichment.csv`.

### `80_atlas_catalogue.py`

Module 6.1 — circadian cell-type atlas resource.

Outputs (4) in `results/80_atlas/`: `celltype_chronotype.csv`, `foreign_marker_summary.csv`, `rhythmic_catalogue.csv`, `rhythmic_marker_genes.csv`.

### `81_cell_cycle.py`

Module 6.3 — rhythmic proliferation / haematopoiesis.

Outputs (2) in `results/81_cell_cycle/`: `cell_cycle_by_sample.csv`, `cell_cycle_rhythm.csv`.

### `82_systemic_cues_readiness.py`

Module 6.4–6.5 — systemic-cue decomposition and time-of-day immune readiness.

Outputs (4) in `results/82_systemic/`: `immune_readiness_rhythm.csv`, `systemic_cue_attribution.csv`, `systemic_cue_common_phase.csv`, `systemic_cue_rhythm.csv`.

### `83_hearing_chronopharm.py`

Module 6.6 — hearing-relevant rhythms and chronopharmacology windows.

Outputs (2) in `results/83_hearing/`: `chronopharmacology_windows.csv`, `hearing_gene_rhythm.csv`.

### `84_gwas_overlap.py`

Module 6.7 — human genetic relevance of rhythmic programmes (GWAS Catalog gene overlap).

Outputs (2) in `results/84_gwas/`: `gwas_enrichment.csv`, `gwas_trait_genes.csv`.

### `97_plan_status.py`

Plan-execution status: every item of the approved plan with its status, producing scripts, main outputs and any deviation (with the reason).

Outputs (1) in `results/`: `plan_execution_status.csv`.

### `98_key_findings.py`

Compile key numbers from all modules into results/key_findings.csv (one row per finding, with the source file it comes from).

Outputs (1) in `results/`: `key_findings.csv`.

### `99_results_manifest.py`

Final catalogue of every deliverable CSV under results/ (path, module, rows, columns, md5, size, producing script, description).

### `200_time_axis_audit.py`

Time-axis audit of every result CSV (ZT = clock − 8; lights on 08:00).

Outputs (2) in `results/r2_audit/`: `time_axis_checks.csv`, `time_axis_columns.csv`.

### `201_figure_replication.py`

Replication of the authors' figure and its extension to every cell type.

Outputs (5) in `results/r2_figure/`: `abundance_pct_rhythm.csv`, `figure_curves.csv`, `figure_phase_relationships.csv`, `figure_points.csv`, `figure_replication_check.csv`.

### `202_abundance_multimodel.R`

Multi-model robustness of every abundance rhythm (cell type × tissue).

Outputs (2) in `results/r2_figure/`: `abundance_multimodel_long.csv`, `abundance_multimodel_summary.csv`.

### `210_cochlea_sample_pattern.py`

The tissue-wide sample pattern (cochlea; BM for comparison).

Outputs (6) in `results/r2_cochlea/`: `gene_pattern_scores_bm.csv`, `gene_pattern_scores_cochlea.csv`, `loto_worst_timepoint.csv`, `sample_pattern_celltypes.csv`, `sample_pattern_covariates.csv`, `sample_pattern_scores.csv`.

### `211_cochlea_relaxed_tier.R`

Calibrated relaxed rhythm tier for the cochlea ("Tier C").

Outputs (4) in `results/r2_cochlea/`: `cochlea_relaxed_calibration.csv`, `cochlea_relaxed_rule_selection.csv`, `cochlea_relaxed_simulation.csv`, `cochlea_relaxed_tier.csv`.

### `212_tierC_downstream.py`

Downstream summaries of the calibrated relaxed cochlear tier (Tier C, 211); exploratory.

Outputs (3) in `results/r2_cochlea/`: `tierC_catalogue.csv`, `tierC_celltype_summary.csv`, `tierC_hearing_windows.csv`.

### `220_external_prepare.py`

Prepare additional external circadian datasets in the format (external/prepared_r2/<dataset>__expr.csv = genes × samples, <dataset>__meta.csv = sample, time_h, time_type, condition, tissue, replicate, value_type, label_note).

Outputs (1) in `results/r2_external/`: `external_prepare_report.csv`.

### `221_external_rhythm.R`

/WP4 — rhythm analysis of the additional external datasets (external/prepared_r2, from 220_external_prepare.py), with the model (71_external_rhythm.R): per dataset × tissue × condition, normcounts → log2(x + 1) and limma-trend, log2 arrays → limma; ~ cos + sin (24 h), robust eBayes, moderated F; mes

Outputs (14) in `results/r2_external/`: `external_dryR_GSE198770_bone_marrow_neutrophil_depleted.csv`, `external_dryR_GSE198770_gut_neutrophil_depleted.csv`, `external_dryR_GSE198770_liver_neutrophil_depleted.csv`, `external_dryR_GSE198770_lung_neutrophil_depleted.csv`, `external_dryR_GSE198770_skin_neutrophil_Bmal1_KO.csv`, `external_dryR_GSE198770_skin_neutrophil_depleted_DT.csv`, `external_manifest_r2.csv`, `external_rhythm_GSE198770_bone_marrow.csv`, `external_rhythm_GSE198770_gut.csv`, `external_rhythm_GSE198770_liver.csv`, `external_rhythm_GSE198770_lung.csv`, `external_rhythm_GSE198770_skin.csv` and 2 more.

### `222_external_validation.py`

Systematic external validation across cell types.

Outputs (7) in `results/r2_external/`: `celltype_validation_dossier.csv`, `external_validation_matrix.csv`, `gene_phase_concordance.csv`, `literature_comparison.csv`, `signature_depletion_effect.csv`, `signature_rhythm.csv`, `signature_scores.csv`.

### `223_skull_marrow_reference.py`

External reference for temporal-bone marrow: Kolabas et al. 2023 (GSE192616; mouse scRNA-seq of skull, vertebra, pelvis, femur, humerus, scapula marrow, meninges and brain; naive / sham / MCAO).

Outputs (4) in `results/r2_external/`: `skull_marrow_composition.csv`, `skull_marrow_label_transfer.csv`, `skull_marrow_neutrophil_states.csv`, `skull_marrow_similarity.csv`.

### `230_clock_alignment.py`

Clock alignment: one internal clock time across our tissues/cell types and all external datasets.

Outputs (2) in `results/r2_phase/`: `clock_alignment_offsets.csv`, `clock_gene_reference_phases.csv`.

### `231_phase_estimates.py`

Harmonised phase estimates and phase comparisons.

Outputs (4) in `results/r2_phase/`: `clock_phase_equality.csv`, `harmonic_12h_summary.csv`, `phase_comparisons.csv`, `phase_estimates_harmonised.csv`.

### `232_phase_ci_coverage.R`

Coverage of the delta-method 95 % phase CI on the data's own noise.

Outputs (1) in `results/r2_phase/`: `phase_ci_coverage.csv`.

### `240_audit.py`

Automated audit of all result CSVs and the register of known gaps.

Outputs (3) in `results/r2_audit/`: `audit_checks.csv`, `audit_global.csv`, `gap_register.csv`.

### `241_validation_matrix.py`

Positive-findings register and multidimensional validation matrix.

Outputs (2) in `results/r2_audit/`: `positive_findings_register.csv`, `validation_matrix.csv`.

### `243_dryr_null_check.R`

Permutation check of the cross-tissue dryR model classes (35): dryR (dryseq, as in 35) re-run for three shared immune cell types with the cochlear time labels relabelled by 3 random non-identity dihedral classes (BM labels kept).

Outputs (1) in `results/r2_audit/`: `dryr_null_check.csv`.

### `244_label_confidence.py`

Sensitivity of key findings to annotation confidence.

Outputs (2) in `results/r2_audit/`: `label_confidence_abundance.csv`, `label_confidence_genes.csv`.

### `245_embedding_sensitivity.py`

Are embedding-based results tied to the label-supervised scANVI embeddings?

Outputs (1) in `results/r2_audit/`: `bm_likeness_scvi_rhythm.csv`.

### `246_neutrophil_timer_replication.py`

Replication of the neutrophil-timer sign result (77): cochlear neutrophil rhythms predicted the blood-neutrophil ZT13−ZT5 change (GSE86619) positively, temporal-BM neutrophils negatively.

Outputs (1) in `results/r2_audit/`: `neutrophil_timer_replication.csv`.

### `250_feasibility.py`

Feasibility of public treatment-response and spatial datasets (user decision: public data only).

Outputs (1) in `results/r2_feasibility/`: `feasibility_treatment_spatial.csv`.

### `251_pilot_noise_bulk.R`

Pilot — public treatment-response data for the cochlea (bulk RNA-seq).

Outputs (3) in `results/r2_pilots/`: `cochlea_tissue_pseudobulk_cosinor.csv`, `pilot_GSE107086_de.csv`, `pilot_GSE312476_de.csv`.

### `252_pilot_noise_linking.py`

Pilot — linking public cochlear noise-response data (bulk and GeoMx spatial) to the atlas.

Outputs (9) in `results/r2_pilots/`: `pilot_collection_time_inference.csv`, `pilot_daynight_rhythm_validation.csv`, `pilot_geomx_genesets.csv`, `pilot_geomx_signatures.csv`, `pilot_night_programme_check.csv`, `pilot_noise_geneset_overlap.csv`, `pilot_noise_kinetics_genesets.csv`, `pilot_noise_signatures.csv`, `pilot_tierC_independent_validation.csv`.

### `300_external_sc_ingest.py`

/WP1 — ingest external cochlear single-cell count matrices into one AnnData per dataset (raw counts, gene symbols as var_names, obs: dataset, sample, condition, group).

Outputs (7) in `results/r3_ingest/`: `gse135913_ingest_report.csv`, `gse167078_ingest_report.csv`, `gse234926_ingest_report.csv`, `gse274279_ingest_report.csv`, `gse283534_ingest_report.csv`, `gse346691_ingest_report.csv`, `ma2025_ingest_report.csv`.

### `301_external_sc_qc.py`

Cell calling and QC of an ingested external dataset (no ambient-RNA correction, as for the atlas).

Outputs (7) in `results/r3_ingest/`: `gse135913_cell_qc.csv`, `gse167078_cell_qc.csv`, `gse234926_cell_qc.csv`, `gse274279_cell_qc.csv`, `gse283534_cell_qc.csv`, `gse346691_cell_qc.csv`, `ma2025_cell_qc.csv`.

### `301b_emptydrops.R`

Cell calling for one raw (unfiltered) external 10x sample with DropletUtils::emptyDrops.

### `302_reference_mapping.py`

Map a QC'd external dataset onto the atlas gold-standard cochlear labels (lib/refmap.py).

Outputs (14) in `results/r3_ingest/`: `gse135913_label_transfer.csv`, `gse135913_marker_check.csv`, `gse167078_label_transfer.csv`, `gse167078_marker_check.csv`, `gse234926_label_transfer.csv`, `gse234926_marker_check.csv`, `gse274279_label_transfer.csv`, `gse274279_marker_check.csv`, `gse283534_label_transfer.csv`, `gse283534_marker_check.csv`, `gse346691_label_transfer.csv`, `gse346691_marker_check.csv` and 2 more.

### `303_train_reference.py`

Train the scArches-ready scANVI reference of the cochlear atlas (lib/refmap.py) once; it is reused to map (CRA019820), GSE167078 and GSE346691 cells onto the gold-standard labels.

Outputs (1) in `results/r3_refmap/`: `reference_model.csv`.

### `310_external_sc_response.py`

Response of the cochlea to noise in external single-cell data, in our label space.

Outputs (9) in `results/r3_gse167078/`, `results/r3_gse346691/`, `results/r3_ma2025/`: `composition_by_sample.csv`, `composition_contrasts.csv`, `composition_rhythm_links.csv`, `composition_by_sample.csv`, `composition_contrasts.csv`, `composition_rhythm_links.csv`, `composition_by_sample.csv`, `composition_contrasts.csv`, `composition_rhythm_links.csv`.

### `311_external_pseudobulk_de.R`

Pseudobulk differential expression per transferred label in external noise datasets (limma-voom).

Outputs (2) in `results/r3_gse346691/`, `results/r3_ma2025/`: `pseudobulk_de.csv`, `pseudobulk_de.csv`.

### `312_ma2025_rhythm_links.py`

(CRA019820): does time-of-day noise sensitivity follow the cochlear rhythms of the atlas?

Outputs (3) in `results/r3_ma2025/`: `nlrp3_axis.csv`, `rhythm_gene_links.csv`, `sampling_phase.csv`.

### `314_ma2025_bulk.py`

Bulk cochlea (CRA019827; quantified and uploaded by the authors): control vs noise × day vs night.

Outputs (4) in `results/r3_ma2025/`: `bulk_de.csv`, `bulk_links.csv`, `bulk_sampling_phase.csv`, `bulk_validation.csv`.

### `315_gse312298_lps_timeofday.py`

LPS at two times of day in sorted bone-marrow and blood neutrophils (GSE312298).

Outputs (4) in `results/r3_response/`: `gse312298_collection_time.csv`, `gse312298_de.csv`, `gse312298_lps_timeofday_links.csv`, `gse312298_validation.csv`.

### `316_ma2025_myeloid_b.py`

(CRA019820) — myeloid primary hypothesis M9 and the exploratory B-lineage add-on.

Outputs (3) in `results/r3_ma2025/`: `b_addon.csv`, `m9_programmes.csv`, `m9_tests.csv`.

### `320_geomx_deconvolution.py`

Deconvolution of the public cochlear GeoMx data (GSE312646; spiral-ganglion-neuron and supporting-cell AOIs, noise vs control) with our cochlear atlas.

Outputs (2) in `results/r3_spatial/`: `geomx_deconvolution_aoi.csv`, `geomx_deconvolution_summary.csv`.

### `321_spatial_dataset_search.py`

Reproducible search for public spatial transcriptomics of the adult cochlea / inner ear / temporal bone and of skull or long-bone marrow (NCBI GEO E-utilities; GSE series only).

Outputs (1) in `results/r3_spatial/`: `spatial_dataset_search.csv`.

### `330_handling_gc_audit.py`

Is the night immediate-early / NF-κB component a handling effect, and which night rhythms are glucocorticoid-dependent?

Outputs (4) in `results/r3_audit/`: `gc_dependence_by_phase.csv`, `gc_response_phases.csv`, `handling_gene_flags.csv`, `handling_sample_effect.csv`.

### `332_cell_level_supplement.py`

Supplementary analyses on the atlas cell-level counts (labels unchanged).

Outputs (4) in `results/r3_audit/`: `cell_clock_phase_by_sample.csv`, `cell_clock_phase_summary.csv`, `immune_substate_shares.csv`, `immune_substates.csv`.

### `333_cell_clock_depth_matched.py`

R1: is the absent single-cell clock tracking of cochlear immune cells a matter of depth?

Outputs (4) in `results/r10_reanalysis/`: `cell_clock_depth_decision.csv`, `cell_clock_depth_matched.csv`, `cell_clock_depth_reference.csv`, `cell_clock_depth_replicates.csv`.

### `341_bm_relaxed_independence.py`

Does the bone-marrow rhythm evidence predict held-out external data?

Outputs (1) in `results/r3_bm/`: `bm_independence_validation.csv`.

### `350_cochlea_immune_oscillations.py`

Cochlea-only synthesis of cell-type oscillations (user decision: bone marrow is the sole source of cochlear immune cells, so BM⇄cochlea egress/return orderings are not tested; intra-cochlear marrow content is part of the cochlear immune compartment → unadjusted shares are primary, marrow-adjusted fi

Outputs (3) in `results/r3_cochlea/`: `cochlea_celltype_oscillations.csv`, `cochlea_immune_curves.csv`, `cochlea_schematic_peaks.csv`.

### `360_conclusions.py`

Concluding synthesis of the whole analysis (rounds 1–3), built only from result tables.

Outputs (2) in `results/`, `results/r3_response/`: `r3_conclusions.csv`, `noise_response_timecourse.csv`.

### `410_design_benchmark.py`

WP-A — how reliable is a one-sample-per-time-point, 6-time-point design? Benchmark of our rhythm pipeline on dense public time series with an independent (held-out) truth, with and without a sample-level nuisance shaped like our cochlear tissue-wide sample pattern.

Outputs (5) in `results/r4_benchmark/`: `benchmark_clock_recovery.csv`, `benchmark_draws.csv`, `benchmark_pattern_flags.csv`, `benchmark_sensitivity.csv`, `benchmark_summary.csv`.

### `420_bulk_rhythm_attribution.py`

WP-B — which cells carry the replicated bulk rhythms? Atlas-based attribution of externally replicated tissue rhythms (cochlea: GSE88954 sham, n = 3 per time point; BM: GSE198770 bone-marrow control, n = 3).

Outputs (5) in `results/r4_attribution/`: `attribution_celltype_enrichment.csv`, `attribution_controls.csv`, `attribution_mechanism_summary.csv`, `attribution_phase_crosscheck.csv`, `bulk_gene_attribution.csv`.

### `430_driver_annotation.py`

WP-D — what drives each rhythm? Driver classes for every rhythmic call, landscape per cell type, enrichment against expression-matched non-rhythmic genes, and the diurnal-vs-circadian evidence status of key findings.

Outputs (5) in `results/r4_drivers/`: `diurnal_circadian_status.csv`, `driver_calls.csv`, `driver_enrichment.csv`, `driver_landscape.csv`, `driver_phase_null.csv`.

### `440_baboon_conservation.py`

WP-C — conservation in a diurnal primate: baboon diurnal transcriptome atlas (GSE98965; Mure et al. 2018, Science; 64 tissues incl. bone marrow 'BOM', one sample every 2 h over 24 h, log2(FPKM + 1)).

Outputs (5) in `results/r4_crossspecies/`: `baboon_bm_clock_genes.csv`, `baboon_bm_gene_concordance.csv`, `baboon_bm_signature_rhythms.csv`, `baboon_clock_offsets.csv`, `baboon_heatshock_phase.csv`.

### `441_human_neutrophils.py`

WP-C — human relevance of the neutrophil rhythms: blood neutrophils of 3 healthy donors at 08:00 and 14:00 (GSE136632, normalised CPM, paired by donor).

Outputs (2) in `results/r4_crossspecies/`: `human_neutrophil_gene_deltas.csv`, `human_neutrophil_timeofday.csv`.

### `450_atlas_concordance.py`

WP-E — positioning of the atlas against independent adult mouse cochlear single-cell / single-nucleus atlases (labels transferred onto our gold standard by 300 → 301 → 302; our labels are never changed).

Outputs (5) in `results/r4_concordance/`: `atlas_concordance_summary.csv`, `atlas_coverage.csv`, `author_label_agreement.csv`, `author_label_concordance.csv`, `composition_similarity.csv`.

### `450b_seurat_meta.R`

WP-E — export the author cell metadata of GSE234926 (P90 cochlear-duct Fbxo2+ nuclei; Seurat object) for label concordance in 450_atlas_concordance.py.

### `451_ageing_rhythm_programmes.py`

WP-E (extended) — are the atlas's rhythmic programmes age-sensitive? GSE274279 snRNA cochlea at 3, 12 and 24 months (24M has two libraries, 24M_Cochlea and 24M_Cochlea_BU); labels transferred onto our gold standard (302).

Outputs (3) in `results/r4_ageing/`: `ageing_composition.csv`, `ageing_rhythmic_variability.csv`, `ageing_sampling_phase.csv`.

### `460_resource_tables.py`

WP-F — submission resource tables (raw CSV) that consolidate every round into one place per entity, with a data dictionary.

Outputs (18) in `results/r4_resource/`: `S_abundance.csv`, `S_bcell_composition.csv`, `S_bcell_immgen_mapping.csv`, `S_bcell_primary_hypotheses.csv`, `S_bcell_timeline.csv`, `S_celltypes.csv`, `S_claims_to_tables.csv`, `S_clock.csv`, `S_flow_cohort.csv`, `S_flow_tests.csv`, `S_gene_rhythms.csv`, `S_myeloid_composition.csv` and 6 more.

### `510_b_composition.py`

WP-B1 — B-lineage composition and developmental timing (H1, H2), BM and cochlea with equal weight.

Outputs (4) in `results/r5_bcell/`: `b_composition_long.csv`, `b_composition_summary.csv`, `b_phase_lags.csv`, `b_primary_composition.csv`.

### `520_b_identity_states.py`

WP-B2 — B-cell identity continuum and state alternation (H2 method 2, H3), BM and cochlea (equal weight).

Outputs (9) in `results/r5_bcell/`: `b_identity_drift_rhythm.csv`, `b_ig_rhythm.csv`, `b_milo_wave.csv`, `b_primary_h3.csv`, `b_pseudotime_by_sample.csv`, `b_pseudotime_checks.csv`, `b_pseudotime_rhythm.csv`, `b_state_axes_by_sample.csv`, `b_state_axes_rhythm.csv`.

### `530_b_programmes_clock.py`

WP-B3 — B-cell function programmes, regulators and the B-cell clock (H4; exploratory programme screen).

Outputs (6) in `results/r5_bcell/`: `b_clock_cell_level.csv`, `b_clock_offsets.csv`, `b_gene_calls_summary.csv`, `b_primary_h4.csv`, `b_programme_rhythm.csv`, `b_tf_activity.csv`.

### `540_b_niche_inputs.py`

WP-B4 — niche and systemic inputs to B cells (H5), BM and cochlea.

Outputs (2) in `results/r5_bcell/`: `b_niche_axes.csv`, `b_primary_h5.csv`.

### `541_nichenet_b.R`

WP-B4 / WP-P — NicheNet ligand activity for the day / night programme of each B stage (H5 agreeing method) and of each partner cell type (receiving side of H7), per tissue (partners: cochlea only).

Outputs (1) in `results/r5_bcell/`: `b_nichenet_ligand_activity.csv`.

### `550_b_partner_signals.py`

WP-P — B ↔ partner signals and spatial adjacency in the cochlea (H7, H8).

Outputs (5) in `results/r5_partners/`: `b_partner_geomx.csv`, `b_partner_geomx_aoi.csv`, `b_partner_global_test.csv`, `b_partner_interactions.csv`, `b_primary_h7_h8.csv`.

### `560_b_immgen.py`

WP-X (part 1) — ImmGen sorted B-lineage populations as an external reference for our B labels.

Outputs (2) in `results/r5_bcell/`: `immgen_b_signatures.csv`, `immgen_label_mapping.csv`.

### `561_b_external.py`

WP-X (part 2) — external data for the B-cell hypotheses (agreeing methods) and the exploratory screen.

Outputs (5) in `results/r5_bcell/`: `b_external_bmal1ko.csv`, `b_external_cryko.csv`, `b_external_igapc.csv`, `b_external_order_tests.csv`, `b_external_signature_rhythm.csv`.

### `570_b_synthesis.py`

WP-S — synthesis of the B-cell round: pre-specified primary hypotheses, multi-method matrix, 24-h timeline and BM-vs-cochlea concordance.

Outputs (4) in `results/r5_bcell/`: `b_crosstissue_concordance.csv`, `b_multimethod_matrix.csv`, `primary_hypotheses.csv`, `timeline_24h.csv`.

### `610_myeloid_refs.py`

WP-M1 — external myeloid reference signatures and label correspondence.

Outputs (3) in `results/r6_myeloid/`: `myeloid_label_mapping.csv`, `myeloid_reference_checks.csv`, `myeloid_reference_signatures.csv`.

### `620_myeloid_composition.py`

WP-M2 — myeloid composition rhythms, BM and cochlea (paired).

Outputs (4) in `results/r6_myeloid/`: `m_composition_long.csv`, `m_composition_paired.csv`, `m_composition_summary.csv`, `m_phase_lags.csv`.

### `630_myeloid_trajectories.py`

WP-M3 — myeloid trajectories and fates (M3 primary; M5 agreeing), BM and cochlea paired.

Outputs (5) in `results/r6_myeloid/`: `m_milo_wave.csv`, `m_primary_m3.csv`, `m_trajectory_by_sample.csv`, `m_trajectory_checks.csv`, `m_trajectory_rhythm.csv`.

### `640_myeloid_states.py`

WP-M4 (part 1) — external myeloid state axes, the paired BM–cochlea contrast (M1, M2, M5, M8 primaries).

Outputs (5) in `results/r6_myeloid/`: `m_axes_by_sample.csv`, `m_axes_rhythm.csv`, `m_m2_external_checks.csv`, `m_paired_contrast.csv`, `m_primary_states.csv`.

### `645_myeloid_function_clock.py`

WP-M4 (part 2) — myeloid functional readiness (M4), myeloid clocks (M7), regulators (exploratory).

Outputs (6) in `results/r6_myeloid/`: `m_clock_cell_level.csv`, `m_clock_offsets.csv`, `m_function_global.csv`, `m_function_rhythm.csv`, `m_primary_function_clock.csv`, `m_tf_activity.csv`.

### `650_myeloid_niche.py`

WP-M5 — retention / release axis (M6) and niche cues.

Outputs (5) in `results/r6_myeloid/`: `m_niche_agreeing.csv`, `m_niche_external.csv`, `m_niche_interactions.csv`, `m_niche_ratio_rhythm.csv`, `m_primary_m6.csv`.

### `651_nichenet_myeloid.R`

WP-M5 — NicheNet ligand activity for the day / night programme of each myeloid receiver (M6 agreeing method: CXCL12 activity percentile; other niche ligands exploratory), per tissue.

Outputs (1) in `results/r6_myeloid/`: `m_nichenet_ligand_activity.csv`.

### `660_myeloid_external.py`

WP-M6 — external replication of the myeloid hypotheses (agreeing methods) and exploratory cross-dataset checks.

Outputs (5) in `results/r6_myeloid/`: `m_external_agreeing.csv`, `m_external_axes_ours.csv`, `m_external_bulk.csv`, `m_external_signatures.csv`, `m_external_timeofday.csv`.

### `670_myeloid_synthesis.py`

WP-M7 — synthesis of the myeloid hypotheses M1–M8 and summary tables.

Outputs (4) in `results/r6_myeloid/`: `m_multimethod_matrix.csv`, `m_paired_summary.csv`, `m_primary_hypotheses.csv`, `m_timeline_24h.csv`.

### `675_timer_split_centred.py`

Follow-up — is the neutrophil-timer split (G02: BM states predict the blood ZT13 − ZT5 change negatively, cochlear states positively) neutrophil-specific or carried by the tissue-wide sample-level pattern?

Outputs (1) in `results/r6_myeloid/`: `m_timer_split_centred.csv`.

### `676_timer_split_verification.py`

Follow-up — verification of the neutrophil-timer split (G02) vs the 'sample-level artefact' reading (675).

Outputs (1) in `results/r6_myeloid/`: `m_timer_split_verification.csv`.

### `677_timer_split_joint_null.py`

R2: one exact test of the neutrophil-timer split (G02).

Outputs (2) in `results/r10_reanalysis/`: `timer_split_joint_null.csv`, `timer_split_joint_null_distribution.csv`.

### `680_exploratory_reassessment.py`

EXPLORATORY reassessment of the pre-specified claims (rounds 5, 6 and Ma 2025) with broader, biologically motivated evidence and relaxed thresholds.

Outputs (5) in `results/r7_exploratory/`: `ex_bulk_composition_replication.csv`, `ex_bulk_noise_timeofday_programmes.csv`, `ex_claim_regrade.csv`, `ex_programme_phase_coherence.csv`, `ex_replication_lines.csv`.

### `700_ma_timing.py`

Step 1 — timing audit and sample-level circadian phase.

Outputs (5) in `results/r8_ma_timing/`: `ma_annotation_check.csv`, `ma_group_phase.csv`, `ma_sample_phase.csv`, `ma_timing_audit.csv`, `ma_timing_resolution.csv`.

### `710_ma_integration.py`

Step 2 — × atlas: composition, cell states, clock activity, signalling and noise responses at the inferred collection phases.

Outputs (8) in `results/r8_ma_integration/`: `ma_clock_activity.csv`, `ma_composition_bulk.csv`, `ma_composition_sc.csv`, `ma_deconvolution_validation.csv`, `ma_noise_response.csv`, `ma_signalling.csv`, `ma_states_sc.csv`, `ma_summary_tests.csv`.

### `720_ma_claim_synthesis.py`

Step 3 — claim-level synthesis of the × atlas integration.

Outputs (3) in `results/r8_ma_integration/`: `ma_claim_lines.csv`, `ma_claim_table.csv`, `ma_resolution_and_limits.csv`.

### `900_figure_data.py`

Figure-ready tables for the main figures (results/r9_figures; double as Source Data).

Outputs (139) in `results/r9_figures/`: `ed10_annotation_check.csv`, `ed10_assumption_tests.csv`, `ed10_deconvolution_validation.csv`, `ed10_library_phase.csv`, `ed10_separations.csv`, `ed1_cell_margin.csv`, `ed1_counts_by_sample.csv`, `ed1_label_support.csv`, `ed2_atlas_columns.csv`, `ed2_atlas_dots.csv`, `ed2_atlas_markers.csv`, `ed2_atlas_tree.csv` and 127 more.

### `901_figure_extract.py`

Status-N extractions from the h5ad for Fig. 1 and ED1 / ED2 (results/r9_figures).

Outputs (6) in `results/r9_figures/`: `ed1_sample_qc_quantiles.csv`, `ed1_time_intermixing.csv`, `f1b_umap_joint.csv`, `f1c_dendrogram_leaves.csv`, `f1c_dendrogram_merges.csv`, `f1c_marker_expression.csv`.

### `902_figure_requests.py`

Coca-fig request files for the main figures, generated from results/r9_figures (no values typed by hand).

### `903_figure_rderive.R`

R-side figure extraction (status R: deterministic re-derivation, asserted against stored fits).

Outputs (2) in `results/r9_figures/`: `f3b_tissue_timecourses.csv`, `f4g_neutrophil_state_genes.csv`.

### `904_code_release.py`

Public release of the study and the code index of the submission.

Outputs (2) in `results/r9_release/`: `local_software_versions.csv`, `scripts_index.csv`.

### `905_number_audit.py`

Writing, number audit of the LaTeX manuscript.

Outputs (2) in `results/r9_text/`: `number_audit.csv`, `number_audit_science.csv`.

### `906_figure_qc.py`

Figure QA for composites, after the reference design system: colour-blind (Machado, Oliveira & Fernandes 2009, severity 1.0; protanopia / deuteranopia / tritanopia) and greyscale simulations of each rendered composite, and the smallest CIE76 colour difference between the palette colours of a figure,

### `907_manuscript_build.py`

Writing, build and check the LaTeX manuscript of the active manuscript folder (config paths.manuscript) against the journal limits in config.yaml (journal).

Outputs (2) in `results/r9_release/`, `results/r9_text/`: `submission_contents.csv`, `manuscript_checks.csv`.

### `908_supplementary_tables.py`

Writing, Supplementary Tables 1-20 and their data dictionary (legends in the active manuscript's supplementary.tex).

Outputs (21) in `results/r9_suppl/`: `Supplementary_Table_01_cell_types.csv`, `Supplementary_Table_02_cell_type_summary.csv`, `Supplementary_Table_03_gene_rhythms.csv`, `Supplementary_Table_04_clock_genes.csv`, `Supplementary_Table_05_composition_rhythms.csv`, `Supplementary_Table_06_hearing_gene_rhythms.csv`, `Supplementary_Table_07_flow_cohort_readouts.csv`, `Supplementary_Table_08_flow_cohort_tests.csv`, `Supplementary_Table_09_evidence_grading.csv`, `Supplementary_Table_10_b_lineage_composition.csv`, `Supplementary_Table_11_b_lineage_immgen.csv`, `Supplementary_Table_12_b_lineage_timeline.csv` and 9 more.

### `909_source_data.py`

Writing, Source Data for the Nature Neuroscience submission.

Outputs (226) in `results/r9_source_data/`: `ED_Fig1_panel_files.csv`, `ED_Fig1a_ed1_sample_qc_quantiles.csv`, `ED_Fig1b_ed1_counts_by_sample.csv`, `ED_Fig1c_ed1_label_support.csv`, `ED_Fig1d_ed1_time_intermixing.csv`, `ED_Fig1e_ed1_cell_margin.csv`, `ED_Fig1e_f1b_umap_joint.csv`, `ED_Fig1f_f4a_immune_composition.csv`, `ED_Fig1f_palette_classes.csv`, `ED_Fig10_panel_files.csv`, `ED_Fig10a_ed10_library_phase.csv`, `ED_Fig10b_ed10_separations.csv` and 214 more.

### `910_fig1_atlas.py`

Fig. 1, built in code as one intricate composite at Nature size (180 x 170 mm, 5-7 pt).

### `911_fig3_heat_shock.py`

Fig. 3, one composite at Nature size (180 x 170 mm, 5-7 pt): a systemic night heat-shock programme follows the activity phase.

### `912_fig5_immune_timetable.py`

Fig. 5: the cochlear immune compartment alternates over the day (180 x 170 mm, 5-7 pt).

### `913_fig6_timer_split.py`

Fig. 6, one composite at Nature size (180 x 170 mm, 5-7 pt): cochlear neutrophils keep the circulating (blood) timetable while marrow neutrophils oppose it.

### `914_fig4_immune_channel.py`

Fig. 4: an immune channel connects the cochlea to the temporal-bone marrow (180 x 170 mm, 5-7 pt).

### `916_fig7_independent_cohort.py`

Fig. 7, one composite at Nature size (180 x 170 mm, 5-7 pt; minimal in-figure text): an independent cohort (Ma et al. 2025) replicates the tissue rhythms and time-of-day noise responses; composition and immune clocks agree only conditionally (bottom row g-i, exploratory; stated in the legend).

### `917_sfig1_flow_gating.py`

Supplementary Fig. 5: flow-cytometry gating strategy and controls (Nature Portfolio reporting summary: 'a figure exemplifying the gating strategy is provided in the Supplementary Information'), 183 mm wide, 5-7 pt.

### `920_ed1_qc.py`

Extended Data Fig. 1 (180 x 170 mm): sampling and quality control (supports Fig. 1).

### `921_ed2_markers.py`

Extended Data Fig. 2 (180 x 170 mm): the full marker atlas (supports Fig. 1e, f).

### `922_ed3_external.py`

Extended Data Fig. 3 (180 x 170 mm): external positioning of the atlas (supports Fig. 1e, f).

### `923_ed4_calibration.py`

Extended Data Fig. 4 (180 x 170 mm): calibration and power of the rhythm calls (R09).

### `924_ed5_clocks.py`

Extended Data Fig. 5 (180 x 214 mm since a-f 180 x 170 mm as in v1): clock detail (supports Fig. 2).

### `925_ed6_programmes.py`

Extended Data Fig. 6 (180 x 170 mm): rhythm programmes (supports Figs 2-4).

### `926_ed7_heat_shock.py`

Extended Data Fig. 7 (180 x 170 mm): heat shock and rhythm drivers (supports Fig. 3).

### `927_ed8_b_lineage.py`

Extended Data Fig. 8 (180 x 170 mm): marrow neighbourhoods and the B lineage (supports Fig. 4).

### `928_ed9_myeloid.py`

Extended Data Fig. 9 (180 x 214 mm since a-g as in v2): myeloid hypotheses and timer robustness (supports Fig. 6).

### `929_ed10_ma2025.py`

Extended Data Fig. 10 (180 x 170 mm): the independent cohort in detail (supports Fig. 6).

### `930_image_panels.py`

Publication panels from the authors' imaging data (data/photos).

Outputs (2) in `results/r9_figures/`: `image_panels_log.csv`, `image_spatial_key.csv`.

### `931_image_quant.py`

Quantification of the authors' confocal images (data/photos; local run, the images are not on the server).

Outputs (4) in `results/r9_figures/`: `f4_parabiont_fields.csv`, `f4_parabiont_objects.csv`, `f5a_cd45_channel_check.csv`, `f5a_fut4_cells.csv`.

### `932_flow_validation.py`

Flow-cytometry validation cohort, analysed exactly as pre-specified in manuscript/flow_validation_plan.md (revision 2, sections 6.2-6.4).

Outputs (10) in `results/r9_figures/`, `results/r9_flow/`: `f4f_flow_shares.csv`, `f5i_flow_markers.csv`, `flow_contours.csv`, `flow_gates.csv`, `flow_outliers.csv`, `flow_plots.csv`, `flow_consistency.csv`, `flow_digestion_controls.csv`, `flow_readouts.csv`, `flow_tests.csv`.

### `933_flow_controls.py`

Writing — flow-cytometry controls on all four processing days, for the Supplementary Fig. 5 legend.

Outputs (1) in `results/r9_flow/`: `flow_fmo_controls.csv`.

### `934_parabiosis_survey.py`

(user request 2026-10-17) — quantitative survey of the full parabiosis image collection: exogenous (partner-derived, GFP+) cells across tissues and after noise.

Outputs (11) in `results/r10_parabiosis/`: `parabiosis_acquisitions.csv`, `parabiosis_detection.csv`, `parabiosis_fut4_cells.csv`, `parabiosis_gain_reference.csv`, `parabiosis_objects.csv`, `parabiosis_perilymph_cells.csv`, `parabiosis_perilymph_fields.csv`, `parabiosis_planes.csv`, `parabiosis_ratio.csv`, `parabiosis_summary.csv`, `parabiosis_tissue_fields.csv`.

### `935_parabiosis_figure.py`

Supplementary Figs 1-3: the parabiosis survey (934), raw images first, then the statistics (the authors asked that the figures show the original images followed by their analyses). 180 mm wide, 5-7 pt.

Outputs (2) in `results/r9_figures/`: `sfig_parabiosis_crop_objects.csv`, `sfig_parabiosis_images.csv`.

### `937_parabiosis_noise.py`

Partner-derived (GFP+) cells in the temporal-bone marrow and cochlea of the one noise-exposed wild-type partner (31 days) compared with the unexposed wild-type partners, from the tables of 934, counting partner-derived cells by the upgraded rule of 960.

Outputs (2) in `results/r10_parabiosis/`: `parabiosis_noise_comparison.csv`, `parabiosis_noise_sections.csv`.

### `938_parabiosis_cell_types.py`

Immune cell types of the 54-day wild-type parabiont: which typed cells of marrow and cochlea are partner-derived (GFP+), for every immunostained cell type, beyond the FUT4+ neutrophils typed by 934 (user: "In addition to neutrophils, we have also captured images of other cell types. Please include t

Outputs (3) in `results/r10_parabiosis/`: `parabiosis_celltype_cells.csv`, `parabiosis_celltype_fields.csv`, `parabiosis_celltype_summary.csv`.

### `939_image_robustness.py`

Robustness of the image quantification behind Fig. 4 and Supplementary Figs 2-6 (user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust").

Outputs (5) in `results/r10_parabiosis/`: `robustness_celltypes.csv`, `robustness_claims.csv`, `robustness_fluids.csv`, `robustness_gfp_bodies.csv`, `robustness_perilymph.csv`.

### `940_flow_gate_robustness.py`

Robustness of the pre-specified flow readouts (Fig. 5) to the fixed gate thresholds (user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust").

Outputs (2) in `results/r9_flow/`: `robustness_flow_gates.csv`, `robustness_flow_summary.csv`.

### `941_trace_verification.py`

Verification of the authors' manual vessel tracing in the 3D renders of Fig. 4 (user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust (or verify that the manual labeling is accurate)").

Outputs (2) in `results/r9_figures/`: `trace_verification.csv`, `trace_verification_null.csv`.

### `942_tissue_assignment_evidence.py`

Objective evidence for the tissue assignments of the parabiosis fields that were made by eye (user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust (or verify that the manual labeling is accurate)"). 934 assigns most fields by their file names and some 

Outputs (2) in `results/r10_parabiosis/`: `tissue_assignment_fields.csv`, `tissue_assignment_summary.csv`.

### `943_verification_gallery.py`

Blinded verification gallery of the image calls behind Fig. 4i and Supplementary Fig. 6 (user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust (or verify that the manual labeling is accurate)"; verification = "Gallery for author check (Recommended)").

Outputs (1) in `results/r10_parabiosis/`: `verification_gallery_items.csv`.

### `944_verification_agreement.py`

Agreement of the image calls with blinded visual labels of the verification sample (943; user: "In fact, you are the one who handles all this—ensuring accuracy and performing manual labeling.").

Outputs (3) in `results/r10_parabiosis/`: `verification_agreement.csv`, `verification_claims.csv`, `verification_items_labelled.csv`.

### `945_celltype_nuclei.py`

Immune cell typing of the 54-day wild-type parabiont anchored on segmented nuclei, replacing the marker-object typing of 938 whose objects were single cells in only about half of the cases (944; user: "Currently, the amount of information in our graph—as well as the number of cell types included—has

Outputs (5) in `results/r10_parabiosis/`: `celltype_nuclei_cells.csv`, `celltype_nuclei_summary.csv`, `verification_nuclei_agreement.csv`, `verification_nuclei_items.csv`, `verification_nuclei_labelled.csv`.

### `946_gfp_body_gallery.py`

Blinded verification sample of the partner-derived (GFP+) bodies of the parabiosis survey (934), the quantity behind the cochlea / marrow ratio of Fig. 4 (user: "In fact, you are the one who handles all this—ensuring accuracy and performing manual labeling.").

Outputs (1) in `results/r10_parabiosis/`: `verification_bodies_items.csv`.

### `947_gfp_body_verification.py`

How many GFP+ bodies of the parabiosis survey (934) are partner-derived cells by eye, and what that does to the cochlea / marrow comparison of Fig. 4 (user: "In fact, you are the one who handles all this—ensuring accuracy and performing manual labeling.").

Outputs (3) in `results/r10_parabiosis/`: `verification_bodies_labelled.csv`, `verification_bodies_ratio.csv`, `verification_bodies_summary.csv`.

### `948_si_imaging.py`

Two Supplementary figures with the raw images behind Fig. 4b-d (180 mm wide, 5-7 pt; user: images "integrated seamlessly within the figures, rather than ... a standalone, large-scale image in the main text"; full-size raw images belong to the Supplement).

### `949_si_robustness.py`

Supplementary figure: robustness of the quantification and blinded verification of the image calls behind Fig. 4 and Fig. 5c (180 x 170 mm, 5-7 pt; user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust (or verify that the manual labeling is accurate)").

### `950_image_inventory.py`

Inventory of the authors' raw image folder $COCH_IMAGES for the 3D vessel and partner-cell analyses (user: "There are a great many images in $COCH_IMAGES that can be used for tasks like vessel tracking; you should analyze as many of them as possible").

Outputs (5) in `results/r10_imaging_3d/`: `archive_members.csv`, `image_inventory.csv`, `image_series.csv`, `ims_provenance.csv`, `ims_provenance_channels.csv`.

### `959_imaging_io.py`

Shared readers and registry for the $COCH_IMAGES analyses (scripts 950-958).

### `960_survey_cells.py`

Partner-derived (GFP+) cells of the parabiosis survey with an upgraded body rule (user: "revert to the previous version, while only incorporating improvements to the layout and an upgraded quantitative algorithm").

Outputs (8) in `results/r10_parabiosis/`: `survey_cells_bodies.csv`, `survey_cells_development.csv`, `survey_cells_ratio.csv`, `survey_cells_summary.csv`, `survey_cells_verified.csv`, `verification_survey_agreement.csv`, `verification_survey_items.csv`, `verification_survey_labelled.csv`.

### `961_celltype_nuclei_v2.py`

Nucleus-anchored immune cell typing, rules v2 (final), with a fresh blinded validation sample. 945 (rules v1) put every typed cell on a segmented nucleus and made the marrow calls largely correct (blinded review: CD45 41/41, CD19 8/8, CD14 7/8, FUT4 11/16 single marker+ cells), but in the cochlear d

Outputs (6) in `results/r10_parabiosis/`: `celltype_v2_cells.csv`, `celltype_v2_development.csv`, `celltype_v2_summary.csv`, `verification_v2_agreement.csv`, `verification_v2_items.csv`, `verification_v2_labelled.csv`.

### `962_upgraded_robustness.py`

Robustness of the two upgraded quantification rules used in Fig. 4g and 4i, computed from the per-object features they store (no image re-reading; the detection steps were swept in 939): 1 survey cell rule (960): local GFP contrast 1.5 / 2 / 3, minimum area 15 / 25 / 40 um2, nucleus share 0 / 0.1 / 

Outputs (3) in `results/r10_parabiosis/`: `robustness_celltype_v2.csv`, `robustness_survey_cells.csv`, `robustness_upgraded_claims.csv`.

### `963_survey_cells_extra.py`

The upgraded survey cell rule of 960 applied to the remaining wild-type tissue fields of the parabiosis survey (vestibular-bone marrow, 54 days; liver, 31 days; temporal-bone marrow of the 5- and 7-day partners, which have no cochlear fields; the reference tissues of the Supplementary tissue figure)

Outputs (1) in `results/r10_parabiosis/`: `survey_cells_bodies_extra.csv`.

### `970_science_figures.py`

The five main figures of the Science Research Article, composed from the panels of the Nature Immunology figures (same data tables, same panel code; nothing is re-analysed) in Science style (COCH_FIG_STYLE=science: upper-case panel letters assigned per composite, American spelling; written to figure

### `975_science_fig2_relabel.py`

Science-style copy of the Nature Immunology Fig. 2 (fig2_local_clocks_composite_v5, assembled with coca-fig, so not drawn by our figure code): the vector PDF is edited in place - the six panel letters a-f become A-F and figure words in British spelling become American (lib.figstyle.american) - by re

### `976_science_convert.py`

Supplementary Materials of the Science version, converted from the approved Nature Immunology sources (manuscript/ni, ni-v0.13: NI text frozen as the source version).

Outputs (1) in `results/r10_science/`: `sm_conversion_log.csv`.

### `977_science_manuscript.py`

Build and check the Science version (manuscript/science).

Outputs (1) in `results/r10_science/`: `science_checks.csv`.

### `978_version_consistency.py`

Consistency of the two versions: every number of the Science text (main text, abstract, captions, Supplementary Text and captions) must occur in the Nature Immunology source (main text, legends, Supplementary Information), and the key claims must use the same numbers and the same terms.

Outputs (1) in `results/r10_science/`: `version_consistency.csv`.

### `979_science_package.py`

Submission package of the Science version (manuscript/science/submission/): Manuscript.pdf main text, references, figure captions and Figs 1-5 (977 build) Supplementary_Materials.pdf Materials and Methods, Supplementary Methods, Supplementary Text, Figs. S1-S22, References (continued numbering), cap

Outputs (1) in `results/r10_science/`: `science_submission_contents.csv`.

## Figures, panels and tables

### Fig. 1: A paired day–night atlas of the cochlea and its temporal-bone marrow.

File: `figures/fig1_atlas_v7.*`; script: `910_fig1_atlas.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Study design (the authors' illustration): collection every 4 h at ZT3-ZT23 (LD 12:12), single-cell RNA sequencing, bioinformatic analysis, validation by confocal imaging and flow cytometry; the table gives the six time points and cells per tissue | `results/r9_figures/f1a_sampling_design.csv` |
| b | 160,643 cells of cochlea and adjacent temporal-bone marrow resolve 35 subtypes in 12 classes (joint UMAP; class hue, subtype lightness; class keylines; boxes mark the zooms in d) | `results/r9_figures/f1b_umap_joint.csv`, `results/r9_figures/palette_celltypes.csv`, `results/r9_figures/palette_classes.csv` |
| c | Each sample on the same embedding (cells of that sample coloured by tissue over all cells): time points intermix within subtypes | `results/r9_figures/f1b_umap_joint.csv` |
| d | Zoomed windows of the joint UMAP (i, lymphoid; ii, myeloid and progenitors) with every subtype named | `results/r9_figures/f1b_umap_joint.csv`, `results/r9_figures/palette_celltypes.csv` |
| e | Each subtype carries its own marker (one gene per subtype; structural \| haematopoietic; dot area = fraction expressing, colour = scaled mean) | `results/r9_figures/f1c_markers_structural.csv`, `results/r9_figures/f1d_markers_haematopoietic.csv` |
| f | Atlas card: subtypes in dendrogram order with class, cells, tissue and ZT shares and marker support in four independent cochlear atlases (per-sample composition rows dropped in v7; see ED1b, ED1f) | `results/r9_figures/ed2_atlas_columns.csv`, `results/r9_figures/ed2_atlas_tree.csv`, `results/r9_figures/ed2_atlas_validation.csv` |
| g | Spatial check: neurons, glia and immune cells sit in their expected cochlear regions (control GeoMx deconvolution; types with q < 0.05) | `results/r9_figures/f1f_spatial_geomx.csv` |

### Fig. 2: Cochlear structural cells keep synchronized, light-entrained clocks.

File: `figures/fig2_local_clocks_composite_v5.*`; script: `902_figure_requests.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Core-clock genes peak in canonical order in every structural cell type, shared with marrow | `results/r9_figures/f2a_clock_phase_map.csv`, `results/r9_figures/f2a_clock_phase_map_plot.csv` |
| b | Each structural cell type's clock sits at the canonical phase (offset ≈ 0 h, tight CIs), like reference organs and independent cochleae | `results/r9_figures/f2c_clock_offsets.csv` |
| c | Single-cell clock phase tracks ZT in structural cells (offset ≈ 0 h) but not in immune cells | `results/r9_figures/f2d_cell_phase_offsets.csv` |
| d | Tracking strength per cell type: structural types track, immune types do not | `results/r9_figures/f2e_tracking_strength.csv` |
| e | Replicated cochlear rhythms are carried by neural, glial, vascular, bone and strial cells | `results/r9_figures/f2h_attribution_celltype.csv` |
| f | Replicated whole-cochlea rhythms are reproduced by within-cell regulation, not composition | `results/r9_figures/f2g_attribution_mechanism.csv` |

### Fig. 3: A systemic night heat-shock programme follows the activity phase.

File: `figures/fig3_heat_shock_v3.*`; script: `911_fig3_heat_shock.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Five systemic-cue programmes peak in the active phase across the cell types of both tissues (amplitude-weighted common peaks ZT14.9-19.6); heat shock is the most phase-concentrated programme in marrow (ZT17.1, resultant 0.97) and peaks at ZT17.3 in cochlea (0.83; feeding 0.89). Cell types share the six samples, so the convergence is descriptive | `results/r9_figures/f3a_systemic_phases.csv`, `results/r9_figures/f3a_systemic_common.csv` |
| b | Whole-tissue pseudobulk: 5 of 10 heat-shock genes are rhythmic in marrow (q < 0.1, peaks ZT16.4-17.9); in cochlea none reaches q < 0.1 (Hsph1 p 0.036 at ZT18.8), so the cochlear night programme is resolved only per cell type (a) | `results/r9_figures/f3b_tissue_timecourses.csv` |
| c | The night heat-shock rise recurs in an independent cochlea cohort (9 / 10 genes up after sham surgery, 10 / 10 after adrenalectomy) whereas immediate-early genes do not rise; in our data immediate-early calls are the most handling-like (53-65 % vs 32-36 % of heat-shock and 33-40 % of NF-κB calls) | `results/r9_figures/f3c_night_external.csv`, `results/r9_figures/f3c_handling_classes.csv` |
| d | Promoter HSE carriers peak at ZT17.8-18.4 in marrow progenitors and monocytes (directional permutation q < 0.05); other motifs are rarely phase-concentrated | `results/r9_figures/f3d_motif_phase.csv` |
| e | Only core-clock (94 % vs 40 %) and heat-shock (84 % vs 34 %) calls peak at their constant-darkness phases above chance; neutrophil-state programmes do not | `results/r9_figures/f3e_clock_output.csv`, `results/r9_figures/f3e_dd_by_type.csv` |
| f | Heat shock peaks at night in mouse (ZT17.6) and by day in baboon (ZT7.8) but at the same internal phase once clocks are aligned (ZT17.0 vs 17.7) | `results/r9_figures/f3f_baboon_heatshock.csv` |
| g | Sample-pattern-linked (36 % in both tissues) and clock-controlled calls (25 % marrow, 20 % cochlea) dominate the Tier-1/2 calls; handling-linked calls are frequent in marrow (17 %, up to 35 % in pre-B cells) but not in cochlea (6 %); core-clock, temperature and glucocorticoid drivers are each <= 1.2 %. Cell types with < 20 calls are not drawn | `results/r9_figures/f3g_driver_landscape.csv` |

### Fig. 4: A marrow-connected, largely host-derived immune compartment in the cochlea.

File: `figures/fig4_immune_channel_v3.*`; script: `914_fig4_immune_channel.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | BM-cochlear immune channel: two 3D surface renders with the authors' manual annotation (teal) of vessels extending from the cochlea, and the same views unannotated | `figures/panels/vessels_view1_annotated.png`, `figures/panels/vessels_view2_annotated.png`, `figures/panels/vessels_view1_original.png`, `figures/panels/vessels_view2_original.png`, `results/r9_figures/image_panels_log.csv` |
| b | Neutrophils (green) along vessels (blue) in the temporal bone beside the cochlea: 3D overview with the zoomed region and the zoom | `figures/panels/marrow_neutrophils_overview.png`, `figures/panels/marrow_neutrophils_zoom.png`, `results/r9_figures/image_panels_log.csv` |
| c | Spatial transcriptomics: deconvolved neutrophil states (band, mature, activated, aged) across a tissue section | `figures/panels/spatial_neutrophil_states.png`, `results/r9_figures/image_spatial_key.csv`, `results/r9_figures/image_panels_log.csv` |
| d | FUT4+ neutrophils in bone-marrow, perilymph and endolymph preparations of wild-type mice (Olympus FV3000, 60x; DAPI, FUT4, CD45, merge; one linear display range per channel; pixel size per field; perilymph CD45 acquired with the 561-nm line at 0 % but a real membrane signal, presence only) and per-cell FUT4 and CD45 over local background (931): FUT4+ (>= 2) in 29/30 marrow, 23/23 perilymph and 56/64 endolymph cells; CD45+ in 29/30, 22/23 and 54/64 | `figures/panels/fut4_bone_marrow_f3_*.png`, `figures/panels/fut4_perilymph_f8_*.png`, `figures/panels/fut4_endolymph_f5_*.png`, `results/r9_figures/image_panels_log.csv`, `results/r9_figures/f5a_fut4_cells.csv`, `results/r9_figures/f5a_cd45_channel_check.csv` |
| e | Parabiosis design: wild-type C57BL/6J mice joined to Lyz2-EGFP partners, one pair per pairing duration; in a wild-type partner GFP+ cells are partner-derived (entered from the shared circulation) and GFP- myeloid cells host-derived | `results/r9_figures/f4d_parabiosis_pairs.csv` |
| f | Raw images: 54-day marrow and cochlear lateral wall of the wild-type partner (GFP, FUT4, DAPI; all 13 lateral-wall FUT4+ cells GFP-negative, asterisks) and perilymph cells of the 28-day wild-type and GFP partners (GFP, DAPI; GFP+ cells circled) | `figures/panels/parabiont_marrow_merge.png`, `figures/panels/parabiont_cochlea_merge.png`, `results/r9_figures/image_panels_log.csv`, `results/r9_figures/f4_parabiont_objects.csv`, `figures/panels/pb_s2b_3.png`, `figures/panels/pb_s2b_4.png`, `results/r9_figures/sfig_parabiosis_images.csv`, `results/r9_figures/sfig_parabiosis_crop_objects.csv` |
| g | Partner-derived cells per mm2 (upgraded body rule, 960) sparser in the cochlea than in the adjacent marrow in all 3 pairs (ratio 0.08-0.59; detection-corrected 0.09-0.59; brightest section 0.06-0.38); label-corrected ratio 14 d 0.03 (0.00-0.08); 31 d 1.49 (0.67-4.39); 54 d 0.04 (0.00-0.14) | `results/r9_figures/f4d_parabiosis_pairs.csv` |
| h | Perilymph GFP+ 6-8% in wild-type vs 55-86% in GFP partners (14 and 28 days) | `results/r9_figures/f4d_perilymph_fields.csv` |
| i | Partner-derived share by cell type, 54 days (nucleus-anchored typing): Cochlea, neutrophils (FUT4+; 60x lateral wall) 0/13 (verified 9/9); Cochlea, leukocytes (CD45+) 0/2 (verified 2/2); Marrow, neutrophils (FUT4+) 5/306 (verified 15/22); Marrow, monocytes (CD14+) 0/17 (verified 12/12); Marrow, leukocytes (CD45+) 14/235 (verified 40/41); Marrow, B cells (CD19+; reporter-negative) 3/80 (verified 17/17) | `results/r9_figures/f4i_celltype_share.csv` |

### Fig. 5: The cochlear immune compartment alternates over the day.

File: `figures/fig5_immune_timetable_v3.*`; script: `912_fig5_immune_timetable.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Day-night timetable: in the cochlea the robust / supported myeloid shares peak in the rest phase (ZT5.2-7.0); in the marrow the robust / supported B shares peak at ZT12.8-20.5; robust shares keep their phase in every core composition model (circular SD of the core peaks <= 0.3 h) and when any time point is left out (LOTO phase SD <= 0.7 h) | `results/r9_figures/f5a_share_peaks.csv`, `results/r9_figures/f4c_composition_models.csv`, `results/r9_figures/f4b_share_fits.csv` |
| b | Cochlear band neutrophils (ZT5.2, robust) and classical monocytes (ZT7.0, supported) peak in the rest phase, GMP (ZT5.8) and aged neutrophils (ZT17.9) are suggestive and naive B (ZT17.1) is not supported; in marrow, naive B (ZT20.5, robust) and transitional B (ZT12.8, supported) shares are timed, mature neutrophils (ZT4.9) suggestive; diamond = replicated in the independent flow cohort (register line; cochlear band neutrophils, classical monocytes, marrow naive B) | `results/r9_figures/f4a_immune_composition.csv`, `results/r9_figures/f4b_share_fits.csv` |
| c | Independent flow cohort: cochlear immature neutrophils ZT17 / ZT5 0.67 (0.52-0.85), cochlear classical monocytes 0.40 (0.30-0.53), marrow recirculating B 1.68 (1.34-2.10); Holm p <= 0.0025: all three primaries replicated; representative contour plots with outliers | `results/r9_figures/f4f_flow_shares.csv`, `results/r9_figures/flow_plots.csv`, `results/r9_figures/flow_contours.csv`, `results/r9_figures/flow_outliers.csv`, `results/r9_figures/flow_gates.csv` |
| d | Label-free waves: along pseudotime, neighbourhoods peak progressively later in the neutrophil lineage (marrow r 0.11, cochlea 0.25) and the marrow B lineage (r 0.29; all neighbourhoods, permutation p <= 0.0012); granulopoiesis and B-development order tests remain suggestive (M03, B02) | `results/r9_figures/f4e_milo_waves.csv` |
| e | Tier-1 genes of aged (649) and activated (465) marrow neutrophils ordered by peak phase: 45 % and 57 % peak in the dark phase (per-sample z of voom log2 CPM; refit asserted equal to the stored fits) | `results/r9_figures/f4g_neutrophil_state_genes.csv` |
| f | Our marrow neutrophil-state rhythms predict the night - day difference of independently sorted marrow neutrophils (GSE312298, PBS, ZT22 vs ZT14): r 0.19 over 644 Tier-1 genes (gene-permutation p 0.0005; bulk-marrow comparator r 0.22) | `results/r9_figures/f4h_gse312298_genes.csv` |
| g | Suggestive primate conservation: 50 % of our marrow Tier-1/2 genes peak within 3 h of baboon spleen (null 39 %, p 0.001) and 38 % of baboon marrow (imputed clock offset; null 29 %, p 0.003) | `results/r9_figures/f4i_baboon_concordance.csv` |

### Fig. 6: Cochlear neutrophils keep the circulating timetable; marrow neutrophils oppose it.

File: `figures/fig6_timer_split_v5.*`; script: `913_fig6_timer_split.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Gene by gene, cochlear aged neutrophils predict the blood-neutrophil ZT13 - ZT5 programme (rho 0.35 over 5,123 genes; 0.62 for the 63 genes p < 0.05 in both) and marrow aged neutrophils its inverse (rho -0.33; -0.63 for 133 genes); diamond = the split is supported by the independent flow cohort (CXCR4, register line) | `results/r9_figures/f5b_timer_genes.csv` |
| b | Positive in cochlear aged / activated / mature neutrophils and negative in marrow aged / activated / mature neutrophils and GMP; band neutrophils null in both tissues; unchanged without stress / immediate-early genes. Single correlations (n = 6) reach exact time-relabelling p 0.067 at best (G53), so specificity (d) and external data (f-h) carry the claim | `results/r9_figures/f5c_timer_concordance.csv` |
| c | Specific to neutrophils: the key states are the most extreme cell types of their tissue in the expected direction (cochlea ranks 1, 2, 3 of 33, exact rank-sum p 0.0002; marrow ranks 1, 4, 10 of 21, p 0.040), which a sample-wide pattern cannot produce | `results/r9_figures/f5d_specificity_rank.csv` |
| d | Opposite halves of the day: the blood-timer score of marrow neutrophils peaks at ZT4.0 (permutation p 0.017) and of cochlear neutrophils at ZT17.5 (p 0.067); paired difference p 0.050 (M02 suggestive) | `results/r9_figures/f5e_timer_score.csv`, `results/r9_figures/f5e_timer_score_fit.csv` |
| e | Not neutrophil identity genes: the split holds in the non-enriched genes (aged neutrophils rho 0.25 cochlea, -0.25 marrow) and is weaker in neutrophil-enriched genes | `results/r9_figures/f5f_enriched_genes.csv` |
| f | Neutrophil-dependent: in independent long-bone marrow the rhythmic genes carry the marrow sign (rho -0.18), lost after neutrophil depletion (-0.03) | `results/r9_figures/f5g_depletion.csv` |
| g | Holds in human blood: cochlear neutrophil fits predict the 14:00 - 08:00 change positively (all states r 0.20) and marrow fits negatively (-0.09); bulk cochlea keeps the cochlear sign (GSE88954 sham, rho 0.43) | `results/r9_figures/f5h_external_lines.csv` |
| h | Independent flow cohort: the CXCR4-hi share of cochlear neutrophils falls at night (odds ratio ZT17 / ZT5 0.29, p 6e-07; marrow 0.91, p 0.70; interaction p 3e-06): S2 replicated; the CD62L-hi share does not change (cochlea 1.20, p 0.35; interaction p 0.21): S1 not replicated | `results/r9_figures/f5i_flow_markers.csv`, `results/r9_figures/flow_plots.csv`, `results/r9_figures/flow_contours.csv`, `results/r9_figures/flow_outliers.csv`, `results/r9_figures/flow_gates.csv` |

### Fig. 7: An independent cohort replicates the tissue rhythms and time-of-day noise responses.

File: `figures/fig7_independent_cohort_v5.*`; script: `916_fig7_independent_cohort.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Clock-based dating of the independent cohort (T2): scRNA controls ZT1.7 and day-noise libraries ZT1.0 share a phase, night-noise libraries sit at ZT13.9; bulk controls ZT3.8 / ZT18.0, noise groups ZT6.8 / ZT15.3 (exposure ZT2-4 / ZT14-16; collection times unreported) | `results/r9_figures/f6a_group_phase.csv`, `results/r9_figures/f6a_design_audit.csv` |
| b | Night-noise libraries were collected 11.2 h (±2.7) apart from day-noise libraries, beyond the 5.5-h upper bound of a noise-induced clock shift: exposure and collection time are inseparable for the night-noise scRNA contrasts (conditional evidence) | `results/r9_figures/f6b_phase_separation.csv` |
| c | 110,912 cohort cells mapped onto our 35 labels (72 % confident) | `results/r9_figures/f6c_label_transfer.csv` |
| d | Time-matched bulk controls replicate our cochlear rhythms: predicted vs observed night - day, r 0.24 over 1,595 non-clock genes (permutation p 0.0005; GSE88954 comparator r 0.44) | `results/r9_figures/f6d_bulk_replication.csv` |
| e | All 10 heat-shock genes are higher at night in untreated cochleae (mean +0.45 log2) and the night elevation is lost after noise (interaction -0.38; night - day after noise +0.06) | `results/r9_figures/f6e_heat_shock_bulk.csv` |
| f | Night noise changes 817 genes (524 up, 293 down; q < 0.05) against 183 after day noise (162 up, 21 down) | `results/r9_figures/f6f_noise_response.csv` |
| g | EXPLORATORY, conditional on no noise x time interaction: shares in night- vs day-noise libraries follow our predicted shares (rho 0.75 absolute, 0.69 immune-relative); time-matched bulk deconvolution rho 0.44 absolute, 0.16 immune-relative (declared follow-up) | `results/r9_figures/f6g_composition_sc.csv` |
| h | EXPLORATORY: clocks of 5 of 8 myeloid types shift with the ~11.5-h collection difference of structural cells (within 4 h), as do pooled neutrophils and monocytes | `results/r9_figures/f6h_clock_tracking.csv` |
| i | EXPLORATORY evidence ladder: lines per claim (supporting / inconclusive / opposing / not testable) and the exploratory grade; B-lineage programme rhythms are opposed (rho -0.70); confirmatory verdicts unchanged | `results/r9_figures/f6i_claim_ladder.csv` |

### Extended Data Fig. 1: Sampling and quality control.

File: `figures/ed1_qc_v2.*`; script: `920_ed1_qc.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Per-sample QC distributions (UMIs, genes, mitochondrial and ribosomal %, doublet score) | `results/r9_figures/ed1_sample_qc_quantiles.csv` |
| b | Every cell type is sampled at every time point in its tissue | `results/r9_figures/ed1_counts_by_sample.csv` |
| c | Label support per cell type (median margin of own over best alternative label) | `results/r9_figures/ed1_label_support.csv` |
| d | Cells of each type intermix across sampling times (kNN, observed / expected) | `results/r9_figures/ed1_time_intermixing.csv` |
| e | Cochlear and marrow cells on the joint embedding; per-cell label-support margin | `results/r9_figures/f1b_umap_joint.csv`, `results/r9_figures/ed1_cell_margin.csv` |
| f | Haematopoietic cells per sample: share of all cells and subtype composition of the haematopoietic cells, cochlea and marrow (moved from main Fig. 4) | `results/r9_figures/f4a_immune_composition.csv`, `results/r9_figures/palette_classes.csv` |

### Extended Data Fig. 2: Marker atlas.

File: `figures/ed2_markers_v1.*`; script: `921_ed2_markers.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Each of the 35 subtypes has its own top-3 markers (fraction expressing, scaled mean) | `results/r9_figures/ed2_top3_markers.csv` |
| b | Lineage markers are specific to their target types | `results/r9_figures/ed2_atlas_markers.csv`, `results/r9_figures/ed2_atlas_dots.csv` |
| c | Label transfer to four external cochlear atlases: transferred cells, confidence and marker check | `results/r9_figures/ed2_atlas_validation.csv` |

### Extended Data Fig. 3: External positioning of the atlas.

File: `figures/ed3_external_v1.*`; script: `922_ed3_external.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Four external cochlear atlases recover our types (confident share, types with >= 20 confident cells, marker check) | `results/r9_figures/ed3_atlas_summary.csv` |
| b | GSE234926 author cell types map to the expected labels (cell-weighted agreement 0.33 strict, 0.77 broad; ARI 0.17) | `results/r9_figures/ed3_author_contingency.csv`, `results/r9_figures/ed3_author_labels.csv` |
| c | Structural composition of external samples vs ours (rho -0.08 to 0.50; none p < 0.05) | `results/r9_figures/ed3_composition_similarity.csv` |
| d | GeoMx: deconvolved proportions per area, neuron vs supporting-cell regions, control vs noise | `results/r9_figures/ed3_geomx_aoi.csv` |
| e | GeoMx control areas: region contrast of every type (SGN 0.49 vs 0.15; Schwann 0.10 vs 0.45) | `results/r9_figures/ed3_geomx_summary.csv` |

### Extended Data Fig. 4: Calibration and power of the rhythm calls.

File: `figures/ed4_calibration_v1.*`; script: `923_ed4_calibration.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Realised FDP and sensitivity of the rhythm rules on held-out dense series: Tier 1-2 FDP 3-14 % vs 32-37 % for nominal p | `results/r9_figures/ed4_calibration.csv` |
| b | Sensitivity grows with fold change and expression | `results/r9_figures/ed4_sensitivity.csv` |
| c | Core-clock genes are recovered as Tier 1-2 calls | `results/r9_figures/ed4_clock_recovery.csv` |
| d | Smallest gene (alpha 0.001, by expression quintile) and share (alpha 0.05) rhythm detectable with 80 % power per cell type (analytic; gene power corrected for voom weights in ) | `results/r9_figures/ed4_gene_power.csv`, `results/r9_figures/ed4_abundance_power.csv` |
| e | Pipeline simulation with injected rhythms on the data's own noise: power, FDP and phase error per rule | `results/r9_figures/ed4_simulation.csv` |

### Extended Data Fig. 5: Clock detail.

File: `figures/ed5_clocks_v2.*`; script: `924_ed5_clocks.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Clock genes in every primary cell type of both tissues (peak, amplitude, permutation p) | `results/r9_figures/ed5_clock_matrix.csv` |
| b | Clock-gene time courses in single cell types (moved from Fig. 2) | `results/r9_figures/f2b_clock_timecourses.csv` |
| c | Each clock gene peaks at one phase across cell types, near its canonical phase | `results/r9_figures/ed5_phase_equality.csv`, `results/r9_figures/ed5_tissue_clock_order.csv` |
| d | Clocks of independent series tell our sampling times (moved from Fig. 2) | `results/r9_figures/f2f_timetelling.csv` |
| e | Clock network order per cell type (CCD z, canonical-order correlation) | `results/r9_figures/ed5_clock_network.csv` |
| f | Same clock phase in cochlea and marrow for shared types | `results/r9_figures/ed5_crosstissue_offsets.csv` |
| g | Single-cell clocks at matched depth: 5 of 12 structural types track at haematopoietic depth (6 untestable), so the structural vs immune contrast is power-limited; among cells with 3-5 clock UMIs 9 of 12 structural and 0 of 10 haematopoietic types track | `results/r9_figures/ed5_clock_depth_matched.csv` |

### Extended Data Fig. 6: Rhythm programmes.

File: `figures/ed6_programmes_v1.*`; script: `925_ed6_programmes.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Rhythm calls per cell type (Tier 1 / 2, pi1) and their peak-time distribution (chronotype) | `results/r9_figures/ed6_rhythm_calls.csv`, `results/r9_figures/ed6_chronotype.csv` |
| b | Robust pathway / TF-activity rhythms per analysis version (primary, depth-matched, centred, both) and their modal phase | `results/r9_figures/ed6_pathway_versions.csv` |
| c | Peak-time distribution of robust pathway rhythms by analysis version | `results/r9_figures/ed6_pathway_peaks.csv` |
| d | TF activity follows TF mRNA by ~1 h in the cochlea (circular mean, Rayleigh) | `results/r9_figures/ed6_tf_lags.csv`, `results/r9_figures/ed6_tf_lag_summary.csv` |
| e | Promoter-motif carriers: phase per cell type vs expected and tissue-level enrichment (NF-kB enriched in both tissues) | `results/r9_figures/ed6_motif_phase.csv`, `results/r9_figures/ed6_motif_enrichment.csv` |

### Extended Data Fig. 7: Heat shock and rhythm drivers.

File: `figures/ed7_heat_shock_v1.*`; script: `926_ed7_heat_shock.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Dissociation-stress sample effect per sample: night-high in both tissues but not rhythmic (cosinor p 0.11-0.13) | `results/r9_figures/ed7_handling_samples.csv` |
| b | Glucocorticoid-response genes peak near ZT16-18 in marrow; night rhythms are not enriched for adrenalectomy-dependent genes (OR 0.64 marrow, 0.75 cochlea) | `results/r9_figures/ed7_gc_phases.csv`, `results/r9_figures/ed7_gc_dependence.csv` |
| c | Driver evidence among rhythmic calls vs background per tissue and cell type | `results/r9_figures/ed7_driver_enrichment.csv` |
| d | Phase-dependent driver flags vs a peak-shuffled null per cell type | `results/r9_figures/ed7_driver_phase_null.csv` |
| e | Baboon heat-shock programme phase in every tissue (raw and clock-aligned) vs mouse | `results/r9_figures/f3f_baboon_heatshock.csv` |
| f | Systemic-cue programme members in the common phase are a small share of each type's rhythmic genes | `results/r9_figures/ed7_systemic_attribution.csv` |

### Extended Data Fig. 8: Marrow neighbourhoods and the B lineage.

File: `figures/ed8_b_lineage_v1.*`; script: `927_ed8_b_lineage.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Rhythmic Milo neighbourhoods per majority cell type (marrow 481 of 3,386; cochlea 266 of 5,034) | `results/r9_figures/ed8_milo_nhoods.csv` |
| b | B-stage share peaks within the B lineage and of the tissue (naive B late night in marrow) | `results/r9_figures/ed8_b_stage_peaks.csv` |
| c | Twelve pre-specified B hypotheses: exact p, q and method lines; none reaches the bar | `results/r9_figures/ed8_b_scorecard.csv`, `results/r9_figures/ed8_b_methods.csv` |
| d | ImmGen correspondence of the B labels | `results/r9_figures/ed8_immgen.csv` |

### Extended Data Fig. 9: Myeloid hypotheses and timer robustness.

File: `figures/ed9_myeloid_v3.*`; script: `928_ed9_myeloid.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Nine pre-specified myeloid hypotheses: none reaches the / bar (M1-M3 partial) | `results/r9_figures/ed9_m_scorecard.csv`, `results/r9_figures/ed9_m_methods.csv` |
| b | Timer split per neutrophil state: raw vs within-sample centred, with and without stress / IEG genes; the raw prediction is largely sample-pattern-implied (G53), centring is not the specificity test (Fig. 6c) | `results/r9_figures/ed9_timer_robustness.csv` |
| c | M3: neutrophil-branch pseudotime per sample (OT test marrow p 0.033) | `results/r9_figures/ed9_m3_pseudotime.csv` |
| d | M1: maturation axis per sample and neutrophil state | `results/r9_figures/ed9_m1_maturation.csv` |
| e | Test design of the neutrophil timer split: blood ZT13 - ZT5 programme vs the change each neutrophil state predicts between the same times (idealised curves; moved from main Fig. 5) | `nan` |
| f | EXPLORATORY: cochlear neutrophils look fresh (Sell / CD62L-high) at night and aged (Cxcr4 / Cxcr2-high) by day (ZT16.9 vs ZT6.2, permutation p 0.083 / 0.017); the marrow Cxcr4 / Cxcr2 ratio is not rhythmic (p 0.68); moved from the former Fig. 5i | `results/r9_figures/f5i_ageing_cycle.csv` |
| g | Cochlear samples differ 4.8-fold in marrow content (marrow-resident precursors / structural cells) with no daily rhythm (p 0.37); against the marrow-resident precursors (sensitivity) the timed cochlear shares keep their phase (shift <= 0.9 h) while p weakens; moved from Fig. 4f | `results/r9_figures/f4d_marrow_content.csv`, `results/r9_figures/f4d_marrow_adjusted.csv` |
| h | The split as one exact test (cochlea and marrow relabelled independently): aged neutrophils p 0.022, pooled neutrophils p 0.099 | `results/r9_figures/ed9_timer_joint_null.csv`, `results/r9_figures/ed9_timer_joint_null_distribution.csv` |

### Extended Data Fig. 10: The independent cohort in detail.

File: `figures/ed10_ma2025_v1.*`; script: `929_ed10_ma2025.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Collection phase of every library under each clock estimator (T2 primary) | `results/r9_figures/ed10_library_phase.csv` |
| b | Group separations under every estimator vs the 5.5-h noise-shift bound | `results/r9_figures/ed10_separations.csv` |
| c | Library labels agree with noise response and clock phase (no corrections) | `results/r9_figures/ed10_annotation_check.csv` |
| d | Bulk deconvolution validation per label (25 of 34 pass) | `results/r9_figures/ed10_deconvolution_validation.csv` |
| e | Summary tests under the eight timing assumptions (exploratory) | `results/r9_figures/ed10_assumption_tests.csv` |

### Supplementary Fig. 1: Check of the vessel tracing between the temporal-bone marrow and the cochlea.

File: `figures/sfig_route_v1.*`; script: `948_si_imaging.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | View 1: traced render, unannotated render and the trace check (vessel ridges of the unannotated render grey; traced centrelines on a ridge green, off a ridge red; broad paint blue) | `figures/panels/vessels_view1_annotated.png`, `figures/panels/vessels_view1_original.png`, `figures/panels/trace_check_view1.png` |
| b | View 2: as a | `figures/panels/vessels_view2_annotated.png`, `figures/panels/vessels_view2_original.png`, `figures/panels/trace_check_view2.png` |
| c | On-ridge share of randomly placed traces (999 per view) and the observed share | `results/r9_figures/simg_trace_null.csv` |
| d | Spatial transcriptomics map of deconvolved neutrophil states (the authors' export) | `figures/panels/spatial_neutrophil_states.png`, `results/r9_figures/image_spatial_key.csv` |

### Supplementary Fig. 2: Partner-derived cells across the tissues of wild-type parabionts.

File: `figures/sfig_parabiosis_tissues_v1.*`; script: `935_parabiosis_figure.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Raw images: one full optical section per animal and tissue of the wild-type partners (marrow, cochlea, vestibular bone, liver), linear display per channel; 64 of 250 acquisitions included | `figures/panels/pb_s1a_*.png`, `results/r9_figures/sfig_parabiosis_images.csv`, `results/r9_figures/pb_acquisitions.csv` |
| b | GFP channel of 100-um crops with the GFP+ bodies kept (yellow) or rejected (grey) by the upgraded rule, marrow and cochlea of the 14-, 31- and 54-day pairs | `figures/panels/pb_s1b_*.png`, `results/r9_figures/sfig_parabiosis_images.csv`, `results/r9_figures/sfig_parabiosis_crop_objects.csv` |
| c | Partner-derived cells per mm2 of tissue per field and pooled per animal (16 marrow, 22 cochlear, 2 vestibular-bone and 3 liver fields; upgraded rule): pooled marrow 41-150, cochlea 12-24, vestibular bone 581, liver 573 | `results/r9_figures/pb_tissue_fields.csv` |
| d | Cochlea vs adjacent marrow per pair: lower in 3 of 3 (exact sign test P = 0.25); observed ratios 0.08-0.59 | `results/r9_figures/pb_pairs.csv` |
| e | Cochlea-to-marrow ratio per pair as observed, detection-corrected (marrow GFP+ bodies detectable at each cochlear field's settings, median 0.95; gain factor 0.46 for 664 vs 700 V), in each acquisition's brightest section (all below 1, 0.06-0.59) and label-corrected with bootstrap 95% CI: 14 d 0.03 (0.00-0.08); 31 d 1.49 (0.67-4.39); 54 d 0.04 (0.00-0.14) | `results/r9_figures/pb_pairs.csv`, `results/r9_figures/pb_detection.csv`, `results/r9_figures/pb_gain_reference.csv` |

### Supplementary Fig. 3: Perilymph cells of wild-type and GFP partners.

File: `figures/sfig_parabiosis_perilymph_v1.*`; script: `935_parabiosis_figure.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Raw images: full fields of perilymph preparations (wild-type and GFP partners, 14 and 28 days), GFP and DAPI, linear display | `figures/panels/pb_s2a_*.png`, `results/r9_figures/sfig_parabiosis_images.csv` |
| b | Cell calls in 150-um crops: nuclei outlined, GFP+ cells marked | `figures/panels/pb_s2b_*.png`, `results/r9_figures/sfig_parabiosis_images.csv`, `results/r9_figures/sfig_parabiosis_crop_objects.csv` |
| c | GFP+ share of nucleated cells per field and pooled: wild-type 5.6-8.5% vs GFP partner 55-86% | `results/r9_figures/pb_perilymph_fields.csv` |
| d | Cytoplasmic-ring GFP over local background per cell (891 cells) with the pre-set rule | `results/r9_figures/pb_perilymph_cells.csv` |
| e | GFP sensitivity of the wild-type fields relative to the same pair's GFP-partner fields (4.7-14.4-fold) | `results/r9_figures/pb_perilymph_fields.csv` |

### Supplementary Fig. 4: Partner-derived cells after noise exposure.

File: `figures/sfig_parabiosis_noise_v1.*`; script: `935_parabiosis_figure.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Raw images (full optical sections, linear display) above the counted GFP+ cells (GFP channel of 100-um crops): marrow and apex of the noise-exposed pair, marrow and spiral lamina of the unexposed 31-day pair, apex of the unexposed 14-day pair | `figures/panels/pb_s3a_*.png`, `results/r9_figures/sfig_parabiosis_images.csv`, `results/r9_figures/sfig_parabiosis_crop_objects.csv` |
| b | GFP+ cells per mm2 per optical section with pooled values and within-animal bootstrap 95% intervals: noise marrow 41, unexposed 31-day marrow 41; noise apex 18, 31-day spiral lamina 24, 14-day apex 12 | `results/r9_figures/pb_noise_sections.csv`, `results/r9_figures/pb_noise_comparison.csv` |
| c | Noise-exposed pair among the unexposed pairs: cochlea 18 (z -0.04) and marrow 41 (z -1.1) GFP+ cells per mm2, inside the 95% prediction intervals; other pairs cochlea 12-24 | `results/r9_figures/pb_noise_tissue_fields.csv`, `results/r9_figures/pb_noise_pairs.csv`, `results/r9_figures/pb_noise_comparison.csv` |
| d | Noise / unexposed density ratios with within-animal bootstrap 95% intervals: marrow, noise / unexposed 31 d 1.01 (0.14-3.50); cochlea, noise apex / unexposed 31 d spiral lamina 0.74 (0.42-1.26); cochlear apex, noise / unexposed 14 d 1.44 (0.88-2.26) | `results/r9_figures/pb_noise_comparison.csv` |
| e | Perilymph: the noise pair's GFP partner 7% GFP+ against 55-86% in the other GFP partners (detection failed in that session; not interpreted) | `results/r9_figures/pb_noise_perilymph_fields.csv` |

### Supplementary Fig. 5: Immune cell types of the 54-day wild-type partner.

File: `figures/sfig_parabiosis_cell_types_v2.*`; script: `935_parabiosis_figure.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Raw images of the 54-day wild-type partner: neutrophils (FUT4), leukocytes (CD45), B cells (CD19) and monocytes (CD14) in marrow and cochlea, with GFP and DAPI | `figures/panels/pb_s4a_*.png`, `results/r9_figures/sfig_parabiosis_images.csv` |
| b | Typed cells (nucleus-anchored) circled by their GFP call: partner-derived, touching GFP only, or GFP-negative; CD3e stained the labyrinth epithelium and is not interpreted | `figures/panels/pb_s4b_*.png`, `results/r9_figures/sfig_parabiosis_images.csv`, `results/r9_figures/sfig_parabiosis_crop_objects.csv`, `results/r9_figures/pb_celltype_cells.csv` |
| c | Partner-derived share: Neutrophils, cochlea (60x) 0/13 (verified 9/9); Neutrophils, cochlea (20x) 0/11 (verified 1/10); Leukocytes, cochlea 0/2 (verified 2/2); Neutrophils, marrow 5/306 (verified 15/22); Monocytes, marrow 0/17 (verified 12/12); Leukocytes, marrow 13/215 (verified 32/32); Leukocytes, vestibular marrow 1/20 (verified 8/9); B cells, marrow (control) 3/80 (verified 17/17) | `results/r9_figures/pb_celltype_share.csv`, `results/r9_figures/pb_celltype_fields.csv` |
| d | Typed cells per mm2 of tissue per cell type, marrow and cochlea (cochlear neutrophils from the 60x lateral wall) | `results/r9_figures/pb_celltype_share.csv` |

### Supplementary Fig. 6: Robustness and blinded verification of the image and flow quantification.

File: `figures/sfig_robustness_v2.*`; script: `949_si_robustness.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | GFP+ body detection: cochlea / marrow below 1 in every pair across threshold, area and solidity | `results/r9_figures/srob_a_gfp_bodies.csv` |
| b | Perilymph ring rule: wild-type share below the GFP partner's at every setting | `results/r9_figures/srob_b_perilymph.csv` |
| c | Fluid positivity: FUT4+ share of perilymph cells >= 91% at every setting; endolymph 47-94% | `results/r9_figures/srob_c_fluids.csv` |
| d | Upgraded survey cell rule: cochlea / marrow below 1 in every pair at 27/27 settings (0.06-0.66) | `results/r9_figures/srob_d_survey_sweep.csv` |
| e | Label-corrected ratios: 14 d 0.03 (0.00-0.08); 31 d 1.49 (0.67-4.39); 31 d noise 0.90 (0.45-2.21); 54 d 0.04 (0.00-0.14) | `results/r9_figures/srob_e_survey_verified.csv` |
| f | Flow gates: P2 and P3 replicated at all 31 gate settings, P1 at 30/31 | `results/r9_figures/srob_f_flow.csv` |
| g | Typing settings: no cochlear leukocyte partner-derived at any of 243 settings; marrow neutrophils < 5% at all; marrow leukocytes above the reporter-negative B-cell background at 159/243 | `results/r9_figures/srob_g_typing_sweep.csv` |
| h | Blinded verification of the nucleus-anchored typing (945 development, 961 validation; 944 for the 60x lateral wall) | `results/r9_figures/srob_h_typing_verified.csv` |

### Supplementary Fig. 7: Flow-cytometry gating strategy and controls.

File: `figures/sfig1_flow_gating_v2.*`; script: `917_sfig1_flow_gating.py`

| Panel | Content | Tables |
| --- | --- | --- |
| a | Gating strategy, cochlea of one mouse (M002, ZT5, cohort 1; complete in both tissues, cochlear neutrophil share closest to the median): 10 contour plots with outliers, every gate with its frequency (% of the plotted parent, equal to the tube counts) | `results/r9_figures/flow_plots.csv`, `results/r9_figures/flow_contours.csv`, `results/r9_figures/flow_outliers.csv`, `results/r9_figures/flow_gates.csv` |
| b | Gating strategy, temporal-bone marrow of one mouse (M002, ZT5, cohort 1; complete in both tissues, cochlear neutrophil share closest to the median): 12 contour plots with outliers, every gate with its frequency (% of the plotted parent, equal to the tube counts) | `results/r9_figures/flow_plots.csv`, `results/r9_figures/flow_contours.csv`, `results/r9_figures/flow_outliers.csv`, `results/r9_figures/flow_gates.csv` |
| c | Controls of one processing day (cohort 1, ZT5): unstained cochlea and the FMO control of every gated marker (CD11b, Ly6C, CD62L, CXCR4, IgD, IgM) with the same gates: all 8 target gates 0 % | `results/r9_figures/flow_plots.csv`, `results/r9_figures/flow_contours.csv`, `results/r9_figures/flow_outliers.csv`, `results/r9_figures/flow_gates.csv` |

