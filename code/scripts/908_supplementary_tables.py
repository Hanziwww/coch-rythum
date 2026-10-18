"""Round 9 writing, phase G — Supplementary Tables 1-20 and their data dictionary (legends in the active
manuscript's supplementary.tex). D134: published numbers follow the first citation in the article (NUMBER maps
the keys below, which keep the v3 numbering: 1->1, 4-7->2-5, 2->6, 18-19->7-8, 20->9, 8-17->10-19, 3->20).

  1    cell-type labels: full name, abbreviation used in the figures, class, label note, cells per tissue and sample
  2    rhythms of hearing-related genes per tissue and cell type, with peak and trough windows (results/83_hearing)
  3    public data sets analysed: accession, repository, species, tissue, assay, design, use, displayed in, reference
  4-19 the resource tables of results/r4_resource (S_*), released through lib/pubclean (internal provenance columns
       dropped and internal references removed from text cells; the 'source' columns of 10, 13 and 16 named script
       outputs and are dropped; agreement rules given as pointers to script documentation are written out):
       4 cell types, 5 gene rhythms, 6 clock genes, 7 composition, 8-11 B lineage, 12-17 myeloid lineage, 18-19 flow
  20   evidence grading of the positive findings (results/r2_audit/positive_findings_register.csv); the one retired
       entry (a schematic superseded by the marrow-source premise) is omitted. Its identifiers are those of the
       'finding' column of Supplementary Table 19.
Every table is written with lib/io.write_csv (manifest) to results/r9_suppl/, with
Supplementary_Tables_data_dictionary.csv: a curated description of every column of every table, asserted complete and
free of internal references. Dataset references were verified against PubMed and GEO (E-utilities); the NeuMap data
were analysed from GSE266518 (Cerezo-Wallis et al., Nature 2025; PMID 41339555), whose young / old experiment
corresponds to GSE200145.
Run: COCH_ROOT=/path/to/coch-rythum python scripts/908_supplementary_tables.py   (local or server; inputs are results only)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import pubclean  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r9_suppl"
TISSUE = {"bm": "temporal-bone marrow", "cochlea": "cochlea"}
S_TABLES = [(4, "S_celltypes", "cell_type_summary", "Cell-type summary: cells per sample, rhythm calls per tier, share of "
             "rhythmic genes, clock metrics, composition rhythm and power"),
            (5, "S_gene_rhythms", "gene_rhythms", "Gene rhythms per tissue and cell type: cosinor fits, exact permutation "
             "statistics, independent tests and evidence tier"),
            (6, "S_clock", "clock_genes", "Core-clock gene fits and clock offsets per cell type"),
            (7, "S_abundance", "composition_rhythms", "Composition rhythms of every cell type in every frame and model, "
             "with the multi-model verdict"),
            (8, "S_bcell_composition", "b_lineage_composition", "B lineage: stage composition per sample"),
            (9, "S_bcell_immgen_mapping", "b_lineage_immgen", "B lineage: correspondence of atlas stages with ImmGen "
             "populations"),
            (10, "S_bcell_timeline", "b_lineage_timeline", "B lineage: peak times along the developmental order"),
            (11, "S_bcell_primary_hypotheses", "b_lineage_hypotheses", "B lineage: the 12 pre-specified hypotheses and "
             "their decisions"),
            (12, "S_myeloid_composition", "myeloid_composition", "Myeloid lineage: composition per sample"),
            (13, "S_myeloid_multimethod", "myeloid_methods", "Myeloid lineage: independent method lines per hypothesis"),
            (14, "S_myeloid_reference_signatures", "myeloid_reference_signatures", "Myeloid lineage: external reference "
             "signatures"),
            (15, "S_myeloid_neumap_signatures", "myeloid_neutrophil_map_signatures", "Myeloid lineage: neutrophil-map "
             "(tissue versus marrow and ageing) signatures"),
            (16, "S_myeloid_timeline", "myeloid_timeline", "Myeloid lineage: peak times of state axes"),
            (17, "S_myeloid_primary_hypotheses", "myeloid_hypotheses", "Myeloid lineage: the nine pre-specified hypotheses "
             "and their decisions"),
            (18, "S_flow_cohort", "flow_cohort_readouts", "Flow-cytometry cohort: per-mouse readouts"),
            (19, "S_flow_tests", "flow_cohort_tests", "Flow-cytometry cohort: all pre-specified tests")]
DROP_SOURCE = {10, 13, 16}

# accession, repository, species, tissue, assay, design, use in this study, displayed in, reference, DOI / URL
DATASETS = [
    ("GSE54650", "GEO", "mouse", "12 organs", "microarray", "constant darkness, every 2 h over 48 h",
     "calibration of rhythm calls; reference clock phases; constant-darkness phases", "Fig. 3e; ED4; ED5c; ED7c,d",
     "Zhang et al., PNAS 2014", "10.1073/pnas.1408886111"),
    ("GSE88954", "GEO", "mouse", "cochlea", "bulk RNA-seq", "sham or adrenalectomy, every 4 h, n = 3",
     "calibration; clock offsets; time-telling; bulk-rhythm attribution; glucocorticoid dependence; dating template",
     "Fig. 2b,e,f; Fig. 6g; Fig. 7a; ED4; ED5d; ED7b; ED10a", "Cederroth et al., Curr. Biol. 2019",
     "10.1016/j.cub.2019.06.057"),
    ("GSE107086", "GEO", "mouse", "cochlea", "bulk RNA-seq", "about ZT5.5 versus ZT17.5, sham or adrenalectomy, no noise",
     "night heat-shock check", "Fig. 3c", "Cederroth et al., Curr. Biol. 2019", "10.1016/j.cub.2019.06.057"),
    ("GSE198770", "GEO", "mouse", "long-bone marrow and other tissues", "bulk RNA-seq",
     "control or neutrophil depletion, every 4 h, n = 3", "marrow clock offset; bulk-rhythm attribution; neutrophil "
     "dependence of the timer split", "Fig. 2b,f; Fig. 6f; ED5d", "Vicanolo et al., Nature 2025",
     "10.1038/s41586-025-08741-5"),
    ("GSE86619", "GEO", "mouse", "blood neutrophils", "bulk RNA-seq", "ZT5 versus ZT13, n = 3",
     "day-night programme of circulating neutrophils", "Fig. 6a-e; ED9b,e", "Adrover et al., Immunity 2019",
     "10.1016/j.immuni.2019.01.002"),
    ("GSE136632", "GEO", "human", "blood neutrophils", "bulk RNA-seq", "08:00 versus 14:00, three donors",
     "human relevance of the timer split", "Fig. 6g", "Adrover et al., Nat. Immunol. 2020", "10.1038/s41590-019-0571-2"),
    ("GSE142754", "GEO", "mouse", "neutrophils of marrow, blood (ZT5, ZT13), lung and spleen", "scRNA-seq", "steady state",
     "blood timer score in single cells; tissue-versus-marrow axis (myeloid hypotheses)", "Fig. 6g (text); ED9a",
     "Ballesteros et al., Cell 2020", "10.1016/j.cell.2020.10.003"),
    ("GSE137540", "GEO", "mouse", "neutrophils of marrow, blood and spleen", "scRNA-seq", "untreated controls",
     "neutrophil maturation axis", "ED9d", "Xie et al., Nat. Immunol. 2020", "10.1038/s41590-020-0736-z"),
    ("GSE312298", "GEO", "mouse", "sorted marrow neutrophils", "bulk RNA-seq", "PBS arms collected at ZT14 and ZT22",
     "replication of the marrow neutrophil-state programmes", "Fig. 5f", "GEO record (2026)",
     "https://identifiers.org/geo:GSE312298"),
    ("GSE98965", "GEO", "baboon", "64 tissues", "RNA-seq", "every 2 h over 24 h",
     "primate heat-shock phase; phase conservation of marrow rhythms", "Fig. 3f; Fig. 5g; ED7e",
     "Mure et al., Science 2018", "10.1126/science.aao0318"),
    ("GSE283534", "GEO", "mouse", "cochlea and vestibule (10 weeks)", "scRNA-seq", "adult",
     "label transfer and marker checks", "Fig. 1f; ED2c; ED3a,c", "GEO record (2025)",
     "https://identifiers.org/geo:GSE283534"),
    ("GSE135913", "GEO", "mouse; human", "cochlea", "scRNA-seq", "developmental series (adult mouse used)",
     "label transfer and marker checks", "Fig. 1f; ED2c; ED3a,c", "GEO record (2021)",
     "https://identifiers.org/geo:GSE135913"),
    ("GSE274279", "GEO", "mouse", "cochlea and utricle", "snRNA-seq", "3, 12 and 24 months",
     "label transfer and marker checks", "Fig. 1f; ED2c; ED3a", "Xia et al., Cell Rep. 2025",
     "10.1016/j.celrep.2025.115781"),
    ("GSE234926", "GEO", "mouse", "cochlear duct (P90)", "snRNA-seq", "Fbxo2+ nuclei", "label transfer; author labels",
     "Fig. 1f; ED2c; ED3a,b", "Eshel et al., Am. J. Hum. Genet. 2024", "10.1016/j.ajhg.2024.01.008"),
    ("GSE312646", "GEO", "mouse", "cochlea", "GeoMx spatial transcriptomics", "control and noise-exposed areas",
     "spatial deconvolution", "Fig. 1g; ED3d,e", "Yu et al., bioRxiv 2025", "10.64898/2025.12.08.693048"),
    ("GSE15907", "GEO", "mouse", "immune cell populations (ImmGen)", "microarray", "sorted populations",
     "B-stage correspondence", "ED8d", "Heng et al., Nat. Immunol. 2008", "10.1038/ni1008-1091"),
    ("GSE109125", "GEO", "mouse", "immune cell populations (ImmGen ULI)", "RNA-seq", "sorted populations",
     "B-stage correspondence", "ED8d", "Yoshida et al., Cell 2019", "10.1016/j.cell.2018.12.036"),
    ("CRA019820", "GSA (CNCB)", "mouse", "cochlea", "scRNA-seq", "controls and 3 days after day or night noise",
     "independent noise cohort", "Fig. 7a-c,g-i; ED10", "Ma et al., Neurosci. Bull. 2026", "10.1007/s12264-025-01440-1"),
    ("CRA019827", "GSA (CNCB)", "mouse", "cochlea", "bulk RNA-seq", "untreated and 1 day after day or night noise, n = 3",
     "independent noise cohort", "Fig. 7a,b,d-f; ED10", "Ma et al., Neurosci. Bull. 2026", "10.1007/s12264-025-01440-1"),
    ("GSE25585", "GEO", "mouse", "peritoneal macrophages", "microarray", "circadian time course",
     "macrophage clock reference (myeloid hypotheses)", "ED9a; Supplementary Tables 13 and 17",
     "Keller et al., PNAS 2009", "10.1073/pnas.0906361106"),
    ("GSE175609", "GEO", "mouse", "intestinal IgA+ plasma cells (B-cell Arntl knockout and controls)", "bulk RNA-seq",
     "ZT0 versus ZT12", "B-clock hypothesis method line", "ED8c", "Penny et al., Sci. Immunol. 2022",
     "10.1126/sciimmunol.abk2541"),
    ("GSE266518", "GEO", "mouse", "neutrophils of multiple tissues (NeuMap)", "scRNA-seq (BD Rhapsody)",
     "young and old mice, multiple tissues (the experiment of GSE200145, analysed here with sample tags); neutrophil "
     "time-stamp experiment (24, 36 and 72 h)", "tissue-versus-marrow, ageing and chronological-age axes (myeloid "
     "hypotheses)", "ED9a; Supplementary Tables 13, 15 and 16", "Cerezo-Wallis et al., Nature 2025",
     "10.1038/s41586-025-09807-0"),
    ("GSE167078", "GEO", "mouse", "cochlear CD45+ cells", "scRNA-seq", "noise exposure",
     "detection of resident-macrophage genes (myeloid hypothesis M8)", "Supplementary Table 14",
     "Milon et al., Cell Rep. 2021", "10.1016/j.celrep.2021.109758"),
    ("GSE346691", "GEO", "mouse", "cochlea", "scRNA-seq", "noise exposure",
     "detection of resident-macrophage genes (myeloid hypothesis M8)", "Supplementary Table 14", "GEO record (2026)",
     "https://identifiers.org/geo:GSE346691"),
]
DATASET_COLS = ["accession", "repository", "species", "tissue", "assay", "design", "use_in_this_study", "displayed_in",
                "reference", "doi_or_url"]

# ------------------------------------------------------------------------------------------------ data dictionary
COMMON = {
    "tissue": "tissue: cochlea or bm (temporal-bone marrow)",
    "cell_type": "cell-type label of the atlas (Supplementary Table 1)",
    "gene": "mouse gene symbol",
    "peak_zt": "peak time of the fitted 24-h rhythm (ZT, h; ZT0 = lights on, 08:00)",
    "peak_clock": "peak time in clock hours (ZT + 8 h)",
    "trough_zt": "trough time of the fitted rhythm (ZT, h)",
    "amp_log2": "cosinor amplitude (log2 expression units)",
    "ptt_fc": "peak-to-trough fold change of the fitted rhythm",
    "AveExpr": "mean expression (log2 counts per million, voom)",
    "peak_ci_low": "lower bound of the 95% parametric-bootstrap interval of the peak time (ZT, h)",
    "peak_ci_high": "upper bound of the 95% parametric-bootstrap interval of the peak time (ZT, h)",
    "amp_ci_low": "lower bound of the 95% parametric-bootstrap interval of the amplitude (log2)",
    "amp_ci_high": "upper bound of the 95% parametric-bootstrap interval of the amplitude (log2)",
    "p": "cosinor P value (limma-voom moderated F-test, 2 degrees of freedom)",
    "q_BH": "Benjamini-Hochberg q value within the cell type",
    "q_perm": "permutation false-discovery rate (pooled null of the 59 non-identity assignments of samples to times)",
    "loto_phase_sd_h": "circular s.d. of the peak time across the six leave-one-time-point-out refits (h)",
    "aligned_phase": "peak time minus the tissue's clock offset from canonical mouse clock phases (h); comparable across "
                     "data sets and species",
    "r_with_sample_pattern_pc1": "Pearson r between the profile over the six samples and the tissue-wide sample pattern "
                                 "(first principal component of sample-level variation)",
    "hearing_sets": "hearing-related gene set(s) containing the gene",
    "peak_window_zt": "window from 3 h before to 3 h after the peak (ZT)",
    "trough_window_zt": "window from 3 h before to 3 h after the trough (ZT)",
    "peak_clock_time": "peak in clock time (ZT + 8 h)",
    "trough_clock_time": "trough in clock time (ZT + 8 h)",
    "tier": "evidence tier: tier1 or tier2 (Methods), tierC (exploratory relaxed tier of the cochlea), exploratory "
            "(P < 0.05 only) or none",
    "human_orthologue": "one-to-one human orthologue (MGI)",
}
DRIVER = {"clock_controlled": "clock-controlled", "composition_linked": "composition-linked", "core_clock": "core clock",
          "glucocorticoid_driven": "glucocorticoid-driven", "handling_linked": "handling-linked",
          "sample_pattern_linked": "sample-pattern-linked", "temperature": "temperature", "unexplained": "unexplained"}
COMPOSITION = {   # Supplementary Tables 8 and 12 (B-lineage and myeloid composition)
    "frame": "composition frame: share_of_tissue (share of all cells of the sample), a within-lineage share (within_B; "
             "within_myeloid, within_neutrophil, within_mono_mac) or ilr_balance (isometric log-ratio balance along the "
             "developmental order)",
    "feature": "cell type (stage) or balance",
    "amplitude": "cosinor amplitude on the logit scale (balances: on the balance scale)",
    "p_ols": "F-test P of the least-squares cosinor on the logit share (or balance)",
    "p_perm": "exact relabelling P of that cosinor (60 assignments of samples to times)",
    "n_core_supporting": "core models supporting the rhythm (exact permutation P <= 0.05 of the logit cosinor, "
                         "quasi-binomial P < 0.05, beta-binomial P < 0.05, centred log-ratio P <= 0.05)",
    "core_models": "the supporting core models",
    "core_peak_circ_sd_h": "circular s.d. of the peak times of the supporting core models (h)",
    "verdict": "robust (>= 3 core models, leave-one-time-point-out phase s.d. < 3 h and core peaks within s.d. < 3 h), "
               "supported (2), fragile (1, or least-squares P < 0.05) or not rhythmic",
    "boot_peak_ci_halfwidth_h": "95th percentile of the absolute deviation of bootstrap peaks from the estimate "
                                "(2,000 multinomial resamplings of cells within each sample; h)",
    "boot_frac_within_3h": "share of bootstrap peaks within 3 h of the estimate",
    "hc_only_p_ols": "F-test P using only cells with a positive label margin (high-confidence labels)",
    "hc_only_peak_shift_h": "peak shift when using only high-confidence cells (h)",
    "marrow_adjusted_resid_peak_zt": "cochlea: peak of the residual rhythm after marrow-content adjustment (ZT, h; "
                                     "Supplementary Note 3)",
    "mean_value": "mean share (or balance) across the six samples",
}
SPECIFIC = {
    1: {"label": "cell-type label of the atlas (reference annotation, used without re-annotation)",
        "abbreviation": "abbreviation used in the figures", "cell_class": "cell class",
        "structural": "TRUE for structural (non-haematopoietic) cell types",
        "label_note": "label with its maturation or ImmGen correspondence where relevant",
        "n_cells_cochlea": "cells in the six cochlear samples", "n_cells_marrow": "cells in the six marrow samples"},
    2: {"tissue": "tissue: cochlea or temporal-bone marrow", "p": "moderated F-test P value",
        "tier": "evidence tier (tier1, tier2 or exploratory)"},
    3: {"accession": "repository accession", "repository": "GEO or the Genome Sequence Archive (CNCB)",
        "species": "species", "tissue": "tissue or cell population", "assay": "assay", "design": "design",
        "use_in_this_study": "how the data were used", "displayed_in": "figures or tables showing the results",
        "reference": "source publication (or GEO record when unpublished)", "doi_or_url": "DOI or URL"},
    4: {"cell_class": "cell class (Supplementary Table 1)",
        "n_cells_total": "cells of this type in the six samples of the tissue",
        "rhythm_status": "inclusion in rhythm analyses: primary (>= 20 cells in each sample) or flagged (10-19 cells in a "
                         "sample); empty, not analysed",
        "n_clock_genes_expressed": "core-clock genes passing the expression filter",
        "n_clock_genes_p05": "core-clock genes with cosinor P < 0.05",
        "clock_phase_offset_h": "clock phase offset: amplitude-weighted circular mean of observed minus canonical peak "
                                "times of the core-clock genes (h)",
        "phase_coherence": "resultant length (0-1) of the core-clock genes' observed-minus-canonical phase differences",
        "median_clock_amp_log2": "median cosinor amplitude of the core-clock genes (log2)",
        "circ_corr_observed_vs_canonical": "circular correlation of observed and canonical peak times of the core-clock "
                                           "genes",
        "p_circ_corr": "permutation P of that correlation (canonical phases shuffled among genes)",
        "CCD": "clock correlation distance to the mouse pan-tissue reference (smaller, closer to a canonical clock)",
        "CCD_p": "permutation P of the clock correlation distance",
        "cell_level_clock_offset_resultant": "consistency (resultant length, 0-1) of the per-sample offset of single-cell "
                                             "clock phases from ZT",
        "cell_level_clock_p_perm": "exact relabelling P of that consistency (60 assignments of samples to times)",
        "n_genes_tested": "genes passing the expression filter (tested for rhythms)",
        "pi1_storey": "estimated share of rhythmic genes (Storey's pi1 from cosinor P values)",
        "n_tier1": "Tier-1 rhythmic genes (Methods)", "n_tier2": "Tier-2 rhythmic genes (Methods)",
        "perm_global_p": "global exact relabelling P: share of the 60 assignments of samples to times with at least as "
                         "many genes at q < 0.10 and fold change >= 1.5 as observed",
        "perm_empirical_fdr": "empirical false-discovery rate of those genes: mean count under the 59 non-identity "
                              "assignments divided by the observed count",
        "n_tierC_exploratory": "exploratory relaxed-tier calls (cochlea; false-discovery rate not controlled)",
        "abundance_peak_zt": "peak of the type's share of tissue cells (24-h least-squares cosinor; ZT, h)",
        "abundance_peak_clock": "peak of that share in clock hours (ZT + 8 h)",
        "abundance_p_ols": "F-test P of that share rhythm",
        "abundance_p_perm": "exact relabelling P of that share rhythm",
        "abundance_multimodel_verdict": "multi-model verdict of the composition rhythm (Supplementary Table 7)"},
    5: {"edgeR_p": "edgeR quasi-likelihood F-test P (independent method)", "rain_p": "RAIN P (independent method)",
        "jtk_p": "JTK_CYCLE P (independent method)",
        "n_celltypes_rhythmic_in_tissue": "cell types of the tissue in which the gene is a Tier-1/2 rhythm",
        "celltype_restricted": "TRUE if Tier 1 in this cell type and P > 0.2 in every other cell type where tested",
        "foreign_marker_of": "cell type of which the gene is a highly specific marker; such rhythms may follow that "
                             "type's content of the sample",
        "foreign_marker_phase_match": "TRUE if the peak lies within 3 h of the abundance peak of that marker's cell type",
        "celltype_empirical_fdr": "calibrated false-discovery rate of the relaxed-tier calls of the cell type (tierC)",
        "rule_selected": "calling rule selected by calibration for the relaxed tier (tierC)",
        "primary_driver": "primary class of the rhythm, by precedence: core clock > temperature > handling-linked > "
                          "composition-linked > glucocorticoid-driven > clock-controlled > sample-pattern-linked > "
                          "unexplained",
        "core_clock": "member of the curated core-clock set",
        "temperature": "member of the heat-shock (HSF1) set or cold-inducible (Cirbp)",
        "handling_linked": "profile follows the cell type's dissociation-stress profile over the six samples (r > 0.8)",
        "composition_linked": "lineage-foreign marker (see foreign_marker_of)",
        "gc_driven": "glucocorticoid-dependent in the adrenalectomy series (GSE88954) or a curated glucocorticoid-response "
                     "gene, with its clock-aligned peak within 5 h of ZT13",
        "gc_dependent_ext": "rhythmic after sham surgery but not after adrenalectomy in GSE88954 (sham P < 0.05, "
                            "adrenalectomy P > 0.2)",
        "clock_controlled": "constant-darkness support and/or a promoter clock element at its expected phase",
        "clock_controlled_strength": "strong (both criteria) or moderate (one)",
        "dd_support": "rhythmic in >= 2 organs in constant darkness (GSE54650; q < 0.05) with the clock-aligned peak "
                      "within 4 h of ours",
        "dd_n_organs": "organs of GSE54650 (constant darkness) in which the gene is rhythmic (q < 0.05)",
        "clock_elements_phase_consistent": "promoter clock elements (Ebox, Eprime, Dbox, RORE) whose expected phase lies "
                                           "within 4 h of the clock-aligned peak",
        "pattern_linked": "profile follows the tissue-wide sample pattern (r > 0.8)",
        "pattern_r": "correlation of the profile with the tissue-wide sample pattern over the six samples",
        "gwas_trait_groups": "GWAS Catalog trait groups (hearing; chronotype or circadian) mapped to the gene "
                             "(P <= 5e-8; one-to-one orthologue)",
        "replicated_bulk_dataset": "independent bulk time series in which the gene's rhythm replicates (GSE88954 sham "
                                   "cochlea; GSE198770 control marrow)",
        "replicated_bulk_aligned_phase": "clock-aligned peak in that series (h)",
        "replicated_bulk_main_contributor": "cell type contributing most to the bulk rhythm in the attribution",
        "replicated_bulk_main_share": "share of the bulk rhythm attributed to that cell type",
        "replicated_bulk_mechanism": "whether our composition-only or regulation-only profile reproduces the bulk "
                                     "rhythm (composition, regulation or not reproduced)",
        "replicated_bulk_phase_diff_h": "signed circular difference of clock-aligned peaks, ours minus the bulk series (h)",
        "baboon_spleen_p": "cosinor P of the orthologue in baboon spleen (GSE98965)",
        "baboon_spleen_aligned_phase": "clock-aligned peak in baboon spleen (own clock offset; h)",
        "baboon_spleen_phase_diff_h": "signed circular difference of clock-aligned peaks, ours minus baboon spleen (h)",
        "baboon_bm_p": "cosinor P of the orthologue in baboon bone marrow (GSE98965)",
        "baboon_bm_aligned_phase": "clock-aligned peak in baboon bone marrow (clock offset imputed; h)",
        "baboon_bm_phase_diff_h": "signed circular difference of clock-aligned peaks, ours minus baboon bone marrow (h)"},
    6: {"status": "inclusion status of the cell type: primary or flagged (Supplementary Table 4)",
        "canonical_zt": "canonical peak time of the gene in mouse tissues (ZT, h)",
        "aligned_minus_canonical_h": "clock-aligned peak minus canonical peak (h)",
        "tissue_offset_h": "clock offset of the tissue (pooled cell types) from canonical phases (h)"},
    7: {"n_cells_total": "cells of this type in the six samples of the tissue",
        "mean_pct": "mean share of tissue cells across the six samples (%)", "mesor_pct": "cosinor mesor (%)",
        "amplitude_pct": "cosinor amplitude (percentage points)", "rel_amplitude": "amplitude divided by the mesor",
        "ptt_ratio": "peak-to-trough ratio, (mesor + amplitude) / (mesor - amplitude)",
        "peak_ci_halfwidth_h": "half-width of the 95% delta-method interval of the peak time (h)",
        "phase_determinate": "TRUE if that half-width is <= 3 h",
        "F": "F statistic of the cosine and sine terms (2 and 3 degrees of freedom)",
        "p_ols": "F-test P of the least-squares cosinor on the share of tissue cells",
        "q_BH": "Benjamini-Hochberg q value within the tissue",
        "p_perm": "exact relabelling P (60 assignments of samples to times)",
        "loto_min_amp_ratio": "smallest amplitude across the leave-one-time-point-out refits, relative to the full fit",
        "loto_max_p": "largest P across the leave-one-time-point-out refits",
        "boot_amp_lo": "lower bound of the 95% bootstrap interval of the amplitude (2,000 multinomial resamplings of cells)",
        "boot_amp_hi": "upper bound of that interval",
        "boot_peak_zt_lo": "lower bound of the 95% bootstrap interval of the peak time (ZT, h)",
        "boot_peak_zt_hi": "upper bound of that interval (ZT, h)",
        "boot_peak_halfwidth_h": "half-width of that interval (h)",
        "boot_frac_p05": "share of bootstrap resamplings with P < 0.05",
        "n_models": "composition models fitted (frames and models; Methods)",
        "n_models_p05": "models with P < 0.05",
        "core_supporting": "core models supporting the rhythm (of five: centred log-ratio with permutation, propeller, "
                           "sccomp, quasi-binomial and beta-binomial)",
        "verdict": "multi-model verdict (robust, supported, fragile or not rhythmic; Methods)",
        "immune": "TRUE for haematopoietic cell types (cochlea)",
        "sensitivity_marrow_adjusted_prob_fc_gt_1p5": "posterior probability that the residual rhythm after "
            "marrow-content adjustment exceeds a 1.5-fold peak-to-trough change (Supplementary Note 3)",
        "sensitivity_marrow_adjusted_peak_zt": "peak of that residual rhythm (ZT, h)"},
    8: COMPOSITION,
    9: {"reference": "ImmGen data set (microarray, GSE15907; RNA-seq, GSE109125)",
        "our_stage": "B-lineage label of the atlas", "immgen_population": "ImmGen sorted population",
        "spearman_r": "Spearman correlation of pseudobulk profiles",
        "is_best": "TRUE for the best-matching population of the stage",
        "expected": "TRUE if the population is the one expected for the stage"},
    10: {"tissue": "tissue: cochlea, bm (temporal-bone marrow) or external (public time series)",
         "layer": "analysis layer: composition frame, neighbourhood wave, state axis, programme, niche interaction or "
                  "external signature",
         "event": "feature that peaks (stage, neighbourhood group, axis, programme, interaction or signature)",
         "p": "rhythm P of the row (exact relabelling P; cosinor P for external series)",
         "ci_halfwidth_h": "half-width of the bootstrap interval of the peak time (composition layers; h)",
         "status": "verdict or role of the row (composition verdict, primary or exploratory analysis, robustness of "
                   "programmes, ligand peak or the external series' time in our marrow)"},
    11: {"hypothesis": "pre-specified hypothesis (label and short name)",
         "primary_statistic": "pre-specified primary statistic", "primary_p": "P value of the primary statistic",
         "peak_zt": "peak of the primary feature, where applicable (ZT, h)",
         "pre_specified_window": "pre-specified window or criterion",
         "in_window": "TRUE if the estimate meets that window or criterion",
         "composition_verdict": "multi-model composition verdict of the feature (Supplementary Table 8)",
         "detail": "supporting details (core models, lag statistic, peaks, leave-one-time-point-out and bootstrap)",
         "shift_h": "B-lineage clock offset minus the tissue clock offset (h)",
         "ci_halfwidth_h": "half-width of its 95% gene-bootstrap interval (h)",
         "determinate": "TRUE if that half-width is <= 3 h",
         "n_axes": "niche signal axes tested", "axes_p_perm": "exact relabelling P of each axis",
         "n_axes_ligand_leads_0_8h": "axes whose ligand peaks 0-8 h before the B-cell response",
         "n_axes_ligand_gene_rhythmic": "axes whose ligand gene is rhythmic",
         "n_axes_nichenet_top10pct": "axes in the top 10% of NicheNet ligand activity",
         "multinichenet_axes_present": "axes among the MultiNicheNet-prioritized interactions",
         "q_BH_primary_set": "Benjamini-Hochberg q across the 12 primary tests",
         "n_methods_agree": "independent methods agreeing", "n_methods": "independent methods evaluated",
         "decision": "supported, partial or not supported (Supplementary Note 1)"},
    12: COMPOSITION,
    13: {"hypothesis": "pre-specified myeloid hypothesis (M1-M9; Extended Data Fig. 9a)",
         "tissue": "tissue: cochlea, bm (temporal-bone marrow) or paired (cochlea versus marrow)",
         "method": "independent method line", "agrees": "TRUE if the method agrees with the hypothesis by its rule",
         "value": "result of the method", "rule": "pre-specified agreement rule"},
    14: {"signature": "external reference signature (study)", "direction": "direction of the gene in the signature",
         "source": "data set and contrast defining the signature",
         "value": "gene statistic in the defining contrast: Spearman rho with pseudotime (maturation), log2 fold change "
                  "ZT13 versus ZT5 (blood timer) or difference of mean log CP10k (tissue versus marrow); empty for "
                  "curated sets"},
    15: {"signature": "neutrophil-map signature (NeuMap, GSE266518)", "direction": "direction of the gene",
         "mean_logcp10k_difference": "difference of mean log CP10k between the contrasted groups",
         "contrast": "contrasted groups", "filter": "cells retained (neutrophil clusters)",
         "n_cells_hi": "cells in the first group of the contrast", "n_cells_lo": "cells in the second group"},
    16: {"tissue": "tissue: cochlea or bm (temporal-bone marrow)",
         "domain": "analysis domain (composition frame, trajectory, programme, state axis or retention and release)",
         "quantity": "rhythmic quantity", "p_perm": "exact relabelling P (60 assignments of samples to times)",
         "evidence": "verdict or additional test result (OT, optimal-transport distribution test; LOTO, phase s.d. of "
                     "the leave-one-time-point-out refits)"},
    17: {"hypothesis": "pre-specified myeloid hypothesis (M1-M9)", "name": "label and short name",
         "tissues": "tissues tested", "p_by_tissue": "primary P per tissue", "combined_p": "combined primary P",
         "combination": "Fisher combination across tissues, or a single test",
         "n_agreeing_methods_best_tissue": "agreeing independent methods in the best tissue",
         "best_tissue": "tissue with the most agreeing methods",
         "agreeing_methods_best_tissue": "those methods (Supplementary Table 13)",
         "n_methods_evaluated": "independent methods evaluated",
         "n_agreeing_any_tissue": "agreeing methods in any tissue",
         "direction_ok": "TRUE if the pre-specified direction is met",
         "direction_detail": "direction check in detail",
         "max_abs_r_sample_pattern": "largest absolute correlation of the primary statistic with the tissue-wide sample "
                                     "pattern",
         "q_BH": "Benjamini-Hochberg q across the nine primary tests",
         "methods_ok": "TRUE if at least two independent methods agree", "q_ok": "TRUE if q < 0.1",
         "decision": "supported, partial or not supported (Supplementary Note 1)", "note": "note on the decision",
         "within_sample_centred_check": "primary statistic after centring within samples (peak and exact P per tissue)"},
    18: {"sample_id": "sample (mouse and tissue)", "mouse_id": "mouse",
         "tissue": "tissue: cochlea or temporal_bone_marrow",
         "cohort": "cohort (1 or 2; each with both time points)",
         "zt": "sampling time (ZT5 = 13:00, ZT17 = 01:00)",
         "trypan_viability_pct": "viability by trypan blue exclusion (%)",
         "exclusions": "pre-specified exclusions applying to this sample",
         "cd45_pct_cells": "CD45+ cells, % of single cells (T1)",
         "cd45neg_pct_cells": "CD45- cells, % of single cells (T1)",
         "neut_pct_cd45": "neutrophils (Ly6G+ SSC-hi), % of CD45+ cells (T1)",
         "imm_frac_pct": "immature neutrophils (Ly6G-int CD11b-int), % of Ly6G+ cells (T2)",
         "P1": "P1, immature neutrophils, % of CD45+ cells (T1 x T2)",
         "cd11b_pct_cells_T2": "CD11b+ cells, % of single cells in tube T2 (consistency check)",
         "P2": "P2, classical monocytes (CD11b+ Ly6C-hi, monocyte scatter), % of CD45+ cells (T3 / T1)",
         "cd11b_pct_cells_T3": "CD11b+ cells, % of single cells in tube T3 (consistency check)",
         "S1_pct": "S1, CD62L-hi cells, % of Ly6G+ cells (T4)",
         "S1_med": "S1, median CD62L of Ly6G+ cells (bead-relative units; T4)",
         "S2_pct": "S2, CXCR4-hi cells, % of Ly6G+ cells (T5)",
         "S2_med": "S2, median CXCR4 of Ly6G+ cells (bead-relative units; T5)",
         "P3": "P3, recirculating B cells (B220-hi IgD+), % of CD45+ cells (T6 / T1)",
         "recirc_frac_pct": "recirculating B cells, % of B220+ cells (T6)",
         "b220_pct_cells_T6": "B220+ cells, % of single cells in tube T6 (consistency check)",
         "b220_pct_cells_T7": "B220+ cells, % of single cells in tube T7 (consistency check)",
         "marrow_content_index": "cochlea: B220-lo IgM- cells (T7) divided by CD45- cells (T1), the marrow-content index",
         "flag_cd11b_T2_T3": "TRUE if the CD11b+ shares of tubes T2 and T3 differ by more than 20% (relative)",
         "flag_b220_T6_T7": "TRUE if the B220+ shares of tubes T6 and T7 differ by more than 20% (relative)"},
    19: {"test": "readout (P1-P3, S1-S2) or digestion control",
         "family": "test family: primary; primary without samples failing the T2-T3 consistency check; sensitivity "
                   "(divided by the marrow-content index); secondary; control",
         "tissue": "tissue of the test: cochlea, temporal_bone_marrow or marrow (temporal-bone marrow), or interaction "
                   "(tissue x time); empty for controls",
         "readout": "readout", "prediction": "pre-specified prediction",
         "finding": "finding tested by the prediction (Supplementary Table 20)", "model": "statistical model",
         "n_zt5": "mice at ZT5", "n_zt17": "mice at ZT17",
         "gmean_zt5": "geometric mean at ZT5", "gmean_zt17": "geometric mean at ZT17",
         "effect": "effect: ZT17 / ZT5 ratio (percentages, medians), odds ratio (% positive) or the cochlea / marrow "
                   "ratio of these (interaction)",
         "effect_scale": "scale of the effect", "ci_lo": "lower bound of the 95% confidence interval of the effect",
         "ci_hi": "upper bound of the 95% confidence interval of the effect", "p": "two-sided P value of the test",
         "p_holm": "Holm-adjusted P across the three primary readouts",
         "verdict": "replicated (P < 0.05 in the predicted direction; Holm-adjusted for primaries), "
                    "direction-consistent, or opposite direction",
         "table": "row type: test or digestion control",
         "cohort": "cohort of the digestion control (1 or 2)", "zt_group": "time-point group of the processing day",
         "marker": "marker of the digestion control",
         "median_bead_relative_FI_digested": "median fluorescence of Ly6G+ marrow cells taken through the cochlear "
                                             "digestion (bead-relative units)",
         "median_bead_relative_FI_undigested": "median fluorescence of undigested Ly6G+ marrow cells (bead-relative units)",
         "target_pct_digested": "% positive (CD62L-hi or CXCR4-hi) of digested Ly6G+ cells",
         "target_pct_undigested": "% positive (CD62L-hi or CXCR4-hi) of undigested Ly6G+ cells",
         "median_ratio_digested_undigested": "ratio of medians, digested / undigested",
         "pct_pos_change_points": "change in % positive, digested minus undigested (percentage points)"},
    20: {"finding": "finding identifier (referred to in Supplementary Table 19)", "category": "kind of finding",
         "tissue": "tissue: cochlea, bm (temporal-bone marrow), both or paired",
         "statement": "the finding as tested", "peak_zt": "peak time, where applicable (ZT, h)",
         "peak_clock": "peak time in clock hours (ZT + 8 h)", "n_dimensions": "evidence dimensions scored",
         "n_supports": "dimensions supporting", "n_partial": "dimensions partially supporting",
         "n_against": "dimensions against", "n_neutral": "dimensions neutral", "n_not_tested": "dimensions not tested",
         "core_status": "status of the core dimensions (model concordance; resampling or false-discovery rate)",
         "verdict": "robust, supported, suggestive or not supported (Supplementary Note 1)",
         "main_caveats": "dimensions that were against or partial, with their numbers"},
}


def describe(n, col):
    spec = SPECIFIC.get(n, {})
    if col in spec:
        return spec[col]
    if col.startswith("n_cells_ZT"):
        return f"cells of this type in the sample at ZT{col.split('ZT')[1]}"
    m = re.match(r"^(cochlea|marrow)_ZT(\d+)$", col)
    if m:
        return f"cells in the {m.group(1)} sample at ZT{m.group(2)}"
    if col.startswith("min_fc_80pct_power_Q"):
        q = col[-1]
        return (f"smallest peak-to-trough fold change detectable with 80% power at alpha = 0.001 for genes in "
                f"expression quintile Q{q} (Q1, lowest)")
    if col.startswith("driver_share_"):
        return f"share of the type's Tier-1/2 rhythms whose primary class is {DRIVER[col[len('driver_share_'):]]}"
    return COMMON.get(col, "")


def st1():
    pal = pd.read_csv(RESULTS / "r9_figures" / "palette_celltypes.csv")
    cnt = pd.read_csv(RESULTS / "00_audit" / "celltype_counts_by_sample.csv")
    cnt["col"] = cnt["tissue"].map({"bm": "marrow", "cochlea": "cochlea"}) + "_ZT" + cnt["zt"].astype(float).astype(int) \
        .astype(str)
    wide = cnt.pivot_table(index="cell_type", columns="col", values="n_cells", aggfunc="sum", fill_value=0)
    cols = [f"{t}_ZT{z}" for t in ("cochlea", "marrow") for z in (3, 7, 11, 15, 19, 23)]
    wide = wide.reindex(columns=cols, fill_value=0).astype(int)
    out = pal.sort_values("order")[["cell_type", "abbr", "cell_class", "structural", "paper_label", "n_cells_cochlea",
                                   "n_cells_bm"]].rename(columns={"cell_type": "label", "abbr": "abbreviation",
                                                                 "paper_label": "label_note", "n_cells_bm": "n_cells_marrow"})
    out = out.merge(wide, left_on="label", right_index=True, how="left")
    assert out[cols].notna().all().all() and (out[cols].sum(1) == out.n_cells_cochlea + out.n_cells_marrow).all()
    return out


def st2():
    h = pd.read_csv(RESULTS / "83_hearing" / "hearing_gene_rhythm.csv")
    w = pd.read_csv(RESULTS / "83_hearing" / "chronopharmacology_windows.csv")
    keys = ["tissue", "cell_type", "gene", "hearing_sets"]
    out = h.merge(w[keys + ["trough_zt", "peak_window_zt", "peak_clock_time", "trough_window_zt", "trough_clock_time"]],
                  on=keys, how="left")
    assert len(out) == len(h)
    out["tissue"] = out["tissue"].map(TISSUE)
    return out


def st20():
    r = pd.read_csv(RESULTS / "r2_audit" / "positive_findings_register.csv")
    r = r[~r["verdict"].astype(str).str.startswith("retired")]
    cols = ["finding_id", "category", "tissue", "statement", "peak_zt", "peak_clock", "n_dimensions", "n_supports",
            "n_partial", "n_against", "n_neutral", "n_not_tested", "core_status", "verdict", "main_caveats"]
    out = r[cols].rename(columns={"finding_id": "finding"})
    return pubclean.clean(out)


# D134: tables numbered by first citation in the article (Nature Portfolio: SI items cited in sequence); the keys
# above keep the v3 numbers and are mapped here at output, references inside the tables included
NUMBER = {1: 1, 4: 2, 5: 3, 6: 4, 7: 5, 2: 6, 18: 7, 19: 8, 20: 9, 3: 20, **{k: k + 2 for k in range(8, 18)}}
REF = re.compile(r"(Supplementary Tables? )(\d+)((?:(?:, | and |-|–)\d+)*)")


def renumber_text(v):
    if not isinstance(v, str):
        return v
    return REF.sub(lambda m: m.group(1) + re.sub(r"\d+", lambda k: str(NUMBER[int(k.group(0))]), m.group(2) + m.group(3)), v)


def main():
    tables = {1: ("cell_types", "atlas cell-type labels, abbreviations used in the figures, class, label note and cells "
                  "per tissue and sample", st1()),
              2: ("hearing_gene_rhythms", "rhythms of hearing-related genes per tissue and cell type with peak and trough "
                  "windows", st2()),
              3: ("public_datasets", "public data sets analysed in this study with their use, displays and references "
                  "(verified via PubMed / GEO)", pd.DataFrame(DATASETS, columns=DATASET_COLS))}
    for n, src, slug, desc in S_TABLES:
        t = pd.read_csv(RESULTS / "r4_resource" / f"{src}.csv", low_memory=False)
        t = pubclean.clean(t, drop=("source",) if n in DROP_SOURCE else ())
        tables[n] = (slug, desc + " (resource table; internal references removed)", t)
    tables[20] = ("evidence_grading", "evidence grading of every positive finding: statement, evidence-dimension counts, "
                  "verdict and caveats", st20())
    dict_rows = []
    for old in sorted(RESULTS.joinpath(MOD).glob("Supplementary_Table_*.csv")):   # clear tables of earlier numberings
        old.unlink()
    for n, (slug, desc, t) in sorted(tables.items(), key=lambda kv: NUMBER[kv[0]]):
        t = t.copy()
        for c in t.columns:
            if t[c].dtype == object:
                t[c] = t[c].map(renumber_text)
        left = pubclean.residual(t)
        assert not left, (n, left[:3])
        k = NUMBER[n]
        write_csv(t, MOD, f"Supplementary_Table_{k:02d}_{slug}.csv", f"Supplementary Table {k}: {desc} (908)",
                  required=list(t.columns[:1]))
        for c in t.columns:
            d = renumber_text(describe(n, c))
            assert d, f"Supplementary Table {k}: no description for column '{c}'"
            dict_rows.append((k, c, d))
    dic = pd.DataFrame(dict_rows, columns=["supplementary_table", "column", "description"])
    assert not pubclean.residual(dic), pubclean.residual(dic)[:3]
    assert not dic["description"].str.contains(r"source table|script|docstring", case=False).any()
    write_csv(dic, MOD, "Supplementary_Tables_data_dictionary.csv", "Data dictionary of Supplementary Tables 1-20: a "
              "curated description of every column (908)", required=["supplementary_table", "column", "description"])
    log(f"Supplementary Tables 1-20 written; dictionary rows {len(dic)}")


if __name__ == "__main__":
    main()
