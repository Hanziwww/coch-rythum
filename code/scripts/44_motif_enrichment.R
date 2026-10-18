#!/usr/bin/env Rscript
# Module 2.9/3d — cis-regulatory motif enrichment (RcisTarget, mm10 500 bp up / 100 bp down promoter rankings,
# motif collection v10 clustered) of rhythmic genes grouped by peak phase, per tissue.
# Gene sets: Tier-1/2 rhythmic genes pooled over cell types of a tissue (a gene enters a phase bin if its median peak
# across the cell types where it is rhythmic falls in the bin), 4 bins of 6 h (ZT0–6, 6–12, 12–18, 18–24); plus
# each cell type's own Tier-1/2 set when it has >= 50 genes. Expectations: E-box (BMAL1/CLOCK) for genes peaking
# around ZT6–10, D-box (DBP/HLF/TEF/NFIL3) ~ZT10–14, RORE (ROR/REV-ERB) ~ZT18–24.
# Outputs (results/44_motifs/): motif_enrichment.csv
# Round 2: env TIERC=1 uses the cochlear calibrated relaxed tier (Tier C, 211) instead of Tier 1/2 (cochlea only)
#   → results/r2_cochlea/motif_enrichment_tierC.csv (exploratory).
suppressPackageStartupMessages({ library(RcisTarget); library(data.table) })
source("scripts/lib/cosinor.R")
TIERC <- Sys.getenv("TIERC", "0") == "1"
out_dir <- file.path(RESULTS, if (TIERC) "r2_cochlea" else "44_motifs"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
res_dir <- file.path(ROOT, "external/resources")
rank_f <- file.path(res_dir, "mm10_500bp_up_100bp_down_v10_clust.rankings.feather")
annot_f <- file.path(res_dir, "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl")
stopifnot(file.exists(rank_f), file.exists(annot_f))
rankings <- importRankings(rank_f)
annot <- importAnnotations(annot_f)
sets <- list()
for (tis in if (TIERC) "cochlea" else c("cochlea", "bm")) {
  d <- if (TIERC) {
    fread(file.path(RESULTS, "r2_cochlea/cochlea_relaxed_tier.csv"),
          select = c("cell_type", "gene", "tierC", "peak_zt"))[tierC == TRUE, .(group = cell_type, gene, tier = "tierC", peak_zt)]
  } else {
    fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", tis)),
          select = c("group", "gene", "tier", "peak_zt"))[tier %in% c("tier1", "tier2")]
  }
  med <- d[, .(peak = (atan2(mean(sin(W24 * peak_zt)), mean(cos(W24 * peak_zt))) %% (2 * pi)) / W24), by = gene]
  med[, bin := cut(peak, c(0, 6, 12, 18, 24), include.lowest = TRUE, labels = c("ZT0-6", "ZT6-12", "ZT12-18", "ZT18-24"))]
  for (b in levels(med$bin)) sets[[sprintf("%s|pooled|%s", tis, b)]] <- med[bin == b, gene]
  for (ct in unique(d$group)) {
    gs <- d[group == ct, gene]
    if (length(gs) >= 50) sets[[sprintf("%s|%s|all_phases", tis, ct)]] <- gs
  }
}
sets <- lapply(sets, function(g) intersect(g, colnames(rankings@rankings)))
sets <- sets[lengths(sets) >= 20]
message(length(sets), " gene sets")
mot <- cisTarget(sets, rankings, motifAnnot = annot, nesThreshold = 3, aucMaxRank = 0.03 * getNumColsInDB(rankings),
                 nCores = as.integer(Sys.getenv("NCORES", "4")))
mot <- as.data.table(mot)
mot[, c("tissue", "set", "phase_bin") := tstrsplit(geneSet, "|", fixed = TRUE)]
keep <- intersect(c("tissue", "set", "phase_bin", "motif", "NES", "AUC", "highlightedTFs", "TF_highConf", "TF_lowConf",
                    "nEnrGenes", "rankAtMax", "enrichedGenes"), names(mot))
fwrite(mot[, ..keep][order(tissue, set, phase_bin, -NES)], file.path(out_dir, if (TIERC) "motif_enrichment_tierC.csv" else "motif_enrichment.csv"))
print(mot[, .(top_motif = motif[which.max(NES)], NES = max(NES), TFs = TF_highConf[which.max(NES)]),
          by = .(tissue, set, phase_bin)])
message("done")
