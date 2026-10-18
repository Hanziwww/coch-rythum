#!/usr/bin/env Rscript
# Round 9 (D67 / D68) — R-side figure extraction (status R: deterministic re-derivation, asserted against stored fits).
# Fig. 3b: tissue-level pseudobulk (all included cell types summed per sample, as in 251 / 211) → the same expression
# filter and limma-voom cosinor → assert that the refit equals the stored tables (r2_pilots/cochlea_tissue_pseudobulk_
# cosinor.csv, r3_bm/bm_tissue_pseudobulk_cosinor.csv) → export the per-sample voom log2 CPM of the heat-shock,
# immediate-early, NF-κB and reference clock genes together with the stored fit (mesor, a, b, p, q).
# Fig. 5e: marrow aged / activated neutrophils → the same per-group fit as 31_rhythm_limma.R → assert equal to the stored
# 33_rhythm_tiers table → per-sample voom log2 CPM of their Tier-1 genes with the stored fit.
# Outputs: results/r9_figures/f3b_tissue_timecourses.csv, f4g_neutrophil_state_genes.csv.
# Run: scripts/setup/rjob.sh scripts/903_figure_rderive.R
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "r9_figures")

SETS <- list(
  "heat shock" = c("Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"),
  "immediate-early" = c("Fos", "Fosb", "Jun", "Junb", "Egr1", "Egr2", "Egr3", "Ier2", "Nr4a1", "Dusp1", "Atf3", "Klf2",
                        "Klf4", "Zfp36"),
  "NF-kB" = c("Nfkbia", "Nfkbiz", "Tnfaip3", "Ier3", "Cxcl2", "Ccl3", "Ccl4", "Nfkbid", "Rel", "Relb"),
  "clock" = c("Dbp", "Nr1d1", "Nr1d2", "Per1", "Per2", "Per3", "Cry1", "Npas2", "Tef", "Ciart"))
STORED <- c(cochlea = "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv", bm = "r3_bm/bm_tissue_pseudobulk_cosinor.csv")

design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
groups <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))
rows <- list()
for (tis in names(STORED)) {                          # 'tis', not 'tissue': inside [.data.table a name equal to a
  PB <- read_pseudobulk(tis)                          # column resolves to the column
  incl <- groups[level == "cell_type" & tissue == tis & status != "excluded"]
  tot <- NULL
  for (g in incl$group) {
    gc <- group_counts(PB, design_dt, tis, g)
    m <- gc$counts
    tot <- if (is.null(tot)) m else tot + m[rownames(tot), ]
    zt <- gc$zt
    sid <- gc$sample_id
  }
  tot <- tot[expr_filter(tot), ]
  fit <- limma_cosinor(tot, zt)
  st <- fread(file.path(RESULTS, STORED[[tis]]))
  stopifnot(setequal(st$gene, fit$table$gene))
  cmp <- merge(fit$table, st, by = "gene", suffixes = c("", ".st"))
  dev <- max(abs(c(cmp$mesor - cmp$mesor.st, cmp$a - cmp$a.st, cmp$b - cmp$b.st)))
  message(sprintf("%s: %d groups, %d genes, max |refit - stored| = %.2e", tis, nrow(incl), nrow(cmp), dev))
  stopifnot(dev < 1e-6)
  y <- calcNormFactors(DGEList(tot))
  v <- voom(y, cosinor_design(zt))
  stopifnot(max(abs(y$samples$norm.factors - fit$norm_factors)) < 1e-12)
  for (s in names(SETS)) {
    for (gn in intersect(SETS[[s]], rownames(v$E))) {   # 'gn': 'gene' is a column of st
      r <- st[gene == gn]
      stopifnot(nrow(r) == 1)
      rows[[length(rows) + 1]] <- data.table(
        tissue = tis, gene = gn, gene_set = s, sample_id = sid, zt = zt, clock = zt_to_clock(zt),
        log2cpm = as.numeric(v$E[gn, ]), mesor = r$mesor, a = r$a, b = r$b, amp_log2 = r$amp_log2,
        peak_zt = r$peak_zt, p = r$p, q_BH = r$q_BH,
        fit_at_zt = r$mesor + r$a * cos(W24 * zt) + r$b * sin(W24 * zt))
    }
  }
}
out <- rbindlist(rows)
out[, `:=`(source_table = paste(STORED, collapse = "; "), finding_id = fifelse(gene_set == "heat shock", "R05", ""),
           transform = paste("voom log2 CPM of the tissue pseudobulk (all included cell types summed per sample, TMM);",
                             "fit = stored limma-voom cosinor, refit asserted equal (max deviation < 1e-6)"))]
fwrite(out, file.path(out_dir, "f3b_tissue_timecourses.csv"))
message(sprintf("wrote %d rows (%d genes)", nrow(out), uniqueN(out[, .(tissue, gene)])))

# ---- Fig. 5e: Tier-1 genes of aged / activated marrow neutrophils
T33 <- fread(file.path(RESULTS, "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv"))
PBbm <- read_pseudobulk("bm")
rows4 <- list()
for (grp in c("Neutrophil (Aged)", "Neutrophil (Activated)")) {   # 'grp': 'group' is a column of T33
  gc <- group_counts(PBbm, design_dt, "bm", grp)
  cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
  fit <- limma_cosinor(cnt, gc$zt)
  st <- T33[group == grp]
  stopifnot(setequal(st$gene, fit$table$gene))
  cmp <- merge(fit$table, st[, .(gene, mesor_st = mesor, a_st = a, b_st = b)], by = "gene")
  dev <- max(abs(c(cmp$mesor - cmp$mesor_st, cmp$a - cmp$a_st, cmp$b - cmp$b_st)))
  message(sprintf("bm %s: %d genes, max |refit - stored| = %.2e", grp, nrow(cmp), dev))
  stopifnot(dev < 1e-6)
  y <- calcNormFactors(DGEList(cnt))
  v <- voom(y, cosinor_design(gc$zt))
  stopifnot(max(abs(y$samples$norm.factors - fit$norm_factors)) < 1e-12)
  t1 <- st[tier == "tier1"]
  E <- v$E[t1$gene, , drop = FALSE]
  long <- data.table(gene = rep(rownames(E), times = ncol(E)), sample_id = rep(gc$sample_id, each = nrow(E)),
                     zt = rep(gc$zt, each = nrow(E)), log2cpm = as.numeric(E))
  long <- merge(long, t1[, .(gene, mesor, a, b, amp_log2, ptt_fc, peak_zt, p, q_BH, q_perm, tier)], by = "gene")
  long[, `:=`(tissue = "bm", group = grp)]
  rows4[[grp]] <- long
}
out4 <- rbindlist(rows4)
stopifnot(uniqueN(out4[group == "Neutrophil (Aged)", gene]) == 649, uniqueN(out4[group == "Neutrophil (Activated)", gene]) == 465)
out4[, `:=`(source_table = "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv; 30_pseudobulk counts", finding_id = "G01",
            transform = paste("voom log2 CPM of the cell-type pseudobulk (TMM); fit = stored limma-voom cosinor,",
                              "refit asserted equal (max deviation < 1e-6); Tier-1 genes only"))]
setcolorder(out4, c("tissue", "group", "gene", "sample_id", "zt"))
fwrite(out4[order(group, peak_zt, gene, zt)], file.path(out_dir, "f4g_neutrophil_state_genes.csv"))
message(sprintf("wrote %d rows (%d genes)", nrow(out4), uniqueN(out4[, .(group, gene)])))
