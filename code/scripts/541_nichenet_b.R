#!/usr/bin/env Rscript
# Round 5, WP-B4 / WP-P — NicheNet ligand activity for the day / night programme of each B stage (H5 agreeing method)
# and of each partner cell type (receiving side of H7), per tissue (partners: cochlea only).
# Gene sets of interest: genes rhythmic in the receiver (33 primary limma p < 0.05, ptt FC >= 1.25) peaking in the
#   active phase (ZT12–24; 'night') or in the rest phase (ZT0–12; 'day'); background = all genes tested in the receiver.
# Potential ligands: ligands of the NicheNet v2 mouse network (lr_network_mouse_21122021) whose receptor is tested in the
#   receiver and that are tested (AveExpr above the tissue median) in at least one other cell type of the tissue.
# nichenetr::predict_ligand_activities → AUPR-corrected and Pearson per ligand; rank and percentile per receiver × set.
# Output (results/r5_bcell/): b_nichenet_ligand_activity.csv
suppressPackageStartupMessages({ library(data.table); library(nichenetr) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "r5_bcell")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
lt <- readRDS(file.path(ROOT, "external/resources/ligand_target_matrix_nsga2r_final_mouse.rds"))
lr <- as.data.table(readRDS(file.path(ROOT, "external/resources/lr_network_mouse_21122021.rds")))
BST <- c("Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)")
PARTNERS <- c("Hair Cell (Outer/Inner)", "Spiral Ganglion Neuron", "Schwann Cell", "Oligodendrocyte", "SV Marginal Cell",
              "SV Intermediate Cell (Functional)", "SV Intermediate Cell (Pigmented)")
res <- list()
for (tis in c("bm", "cochlea")) {
  g <- fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", tis)),
             select = c("group", "status", "gene", "AveExpr", "p", "ptt_fc", "peak_zt"))
  med <- median(g$AveExpr, na.rm = TRUE)
  receivers <- intersect(c(BST, if (tis == "cochlea") PARTNERS), unique(g$group))
  for (rc in receivers) {
    r <- g[group == rc]
    others <- unique(g[group != rc & AveExpr > med, gene])
    pot <- unique(lr[to %in% r$gene & from %in% others, from])
    pot <- intersect(pot, colnames(lt))
    bg <- intersect(r$gene, rownames(lt))
    for (set in c("night", "day")) {
      gs <- r[p < 0.05 & ptt_fc >= 1.25 & ((set == "night" & peak_zt >= 12) | (set == "day" & peak_zt < 12)), gene]
      gs <- intersect(gs, rownames(lt))
      if (length(gs) < 20 || length(pot) < 20) next
      la <- predict_ligand_activities(geneset = gs, background_expressed_genes = bg, ligand_target_matrix = lt,
                                      potential_ligands = pot)
      la <- as.data.table(la)[order(-aupr_corrected)]
      la[, `:=`(rank = seq_len(.N), percentile = 1 - (seq_len(.N) - 1) / .N)]
      res[[length(res) + 1]] <- la[, .(tissue = tis, receiver = rc, receiver_class = if (rc %in% BST) "B stage" else "partner",
                                        gene_set = set, n_geneset = length(gs), n_background = length(bg), n_potential_ligands = length(pot),
                                        ligand = test_ligand, aupr_corrected, pearson, rank, percentile)]
      message(sprintf("%s %s %s: %d genes, %d ligands", tis, rc, set, length(gs), length(pot)))
    }
  }
}
res <- rbindlist(res)
fwrite(res, file.path(out_dir, "b_nichenet_ligand_activity.csv"))
message("done")
