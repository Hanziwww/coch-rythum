#!/usr/bin/env Rscript
# Module 2.2/2.4 — primary gene-rhythm model per tissue × group (limma-voom cosinor, LimoRhyde framework).
#
# Per included tissue × group (pseudobulk, 6 samples ordered by ZT):
#   expression filter -> TMM -> voom(~ cos + sin) -> lmFit -> eBayes(robust) -> moderated F (2 df)
#   mesor, a, b, amplitude (log2), peak-to-trough FC, peak ZT, parametric-bootstrap CIs (t, df.total)
#   permutation null: refit under all 59 non-identity dihedral relabeling classes -> per-gene class p,
#     pooled-null empirical p and FDR (q_perm), and hit counts per class for a group-level global test
#   LOTO: 6 refits leaving one ZT out -> phase circular SD, min amplitude ratio, max p, worst sample
#   edgeR quasi-likelihood F test (NB GLM) as an independent count-based check
# Usage: Rscript scripts/31_rhythm_limma.R [cell_type|cell_class]
suppressPackageStartupMessages({
  library(limma); library(edgeR); library(data.table); library(BiocParallel)
})
source("scripts/lib/cosinor.R")

args <- commandArgs(trailingOnly = TRUE)
LEVEL <- if (length(args)) args[1] else "cell_type"
NCORES <- as.integer(Sys.getenv("NCORES", "48"))
out_dir <- file.path(RESULTS, "31_rhythm_genes"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == LEVEL & status != "excluded"]
if (nzchar(Sys.getenv("MAXG"))) incl <- incl[seq_len(as.integer(Sys.getenv("MAXG")))]   # smoke tests
if (nzchar(Sys.getenv("MAXG"))) out_dir <- file.path(CACHE, "31_smoke")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
classes <- dihedral_classes(6)
PB <- list(cochlea = read_pseudobulk("cochlea", LEVEL), bm = read_pseudobulk("bm", LEVEL))
message(sprintf("%s: %d groups, %d relabeling classes, %d cores", LEVEL, nrow(incl), length(classes), NCORES))

design_overlap <- function(zt, perm) {
  Xt <- scale(cosinor_design(zt)[, 2:3], scale = FALSE)
  Xp <- scale(cosinor_design(zt[perm + 1])[, 2:3], scale = FALSE)
  s <- svd(crossprod(qr.Q(qr(Xt)), qr.Q(qr(Xp))))$d
  mean(s^2)
}

thresholds <- function(tab) {
  data.table(n_p001 = sum(tab$p < 0.001), n_p01 = sum(tab$p < 0.01), n_q05 = sum(tab$q_BH < 0.05),
             n_q10 = sum(tab$q_BH < 0.10),
             n_q10_fc15 = sum(tab$q_BH < 0.10 & 2^(2 * sqrt(tab$a^2 + tab$b^2)) >= CFG$rhythm$fc_threshold))
}

run_group <- function(i) {
  tissue <- incl$tissue[i]; group <- incl$group[i]
  gc <- group_counts(PB[[tissue]], design_dt, tissue, group, LEVEL)
  keep <- expr_filter(gc$counts)
  cnt <- gc$counts[keep, , drop = FALSE]
  zt <- gc$zt
  res <- limma_cosinor(cnt, zt)
  tab <- res$table
  ap <- amp_phase(tab$a, tab$b)
  tab[, `:=`(amp_log2 = ap$amp, ptt_fc = 2^(2 * ap$amp), peak_zt = ap$peak)]
  tab <- cbind(tab, boot_ci(tab$a, tab$b, tab$se_a, tab$se_b, tab$df_total, nboot = CFG$rhythm$n_boot, seed = i))

  # ---- permutation null over the 59 non-identity relabeling classes
  nullF <- matrix(NA_real_, nrow(tab), length(classes) - 1)
  hits <- vector("list", length(classes))
  hits[[1]] <- cbind(data.table(class = 1L, overlap = 1), thresholds(tab))
  for (k in 2:length(classes)) {
    r <- limma_cosinor(cnt, zt[classes[[k]] + 1], norm_factors = res$norm_factors)$table
    nullF[, k - 1] <- r$F
    hits[[k]] <- cbind(data.table(class = k, overlap = design_overlap(zt, classes[[k]])), thresholds(r))
  }
  tab[, p_perm_class := (1 + rowSums(nullF >= F)) / length(classes)]
  pooled <- sort(as.vector(nullF))                       # ascending
  n_null_ge <- length(pooled) - findInterval(tab$F, pooled, left.open = TRUE)   # #null F >= observed F
  tab[, p_perm_pooled := (1 + n_null_ge) / (1 + length(pooled))]
  # empirical FDR: expected null hits per class at threshold F / observed hits at F, monotonised
  o <- order(tab$F, decreasing = TRUE)
  exp_false <- n_null_ge[o] / (length(classes) - 1)
  fdr <- pmin(1, exp_false / seq_along(o))
  fdr <- rev(cummin(rev(fdr)))
  tab[o, q_perm := fdr]

  # ---- leave-one-timepoint-out
  lo_peak <- lo_amp <- lo_p <- matrix(NA_real_, nrow(tab), length(zt))
  for (j in seq_along(zt)) {
    r <- limma_cosinor(cnt[, -j, drop = FALSE], zt[-j], norm_factors = res$norm_factors[-j])$table
    ap2 <- amp_phase(r$a, r$b)
    lo_peak[, j] <- ap2$peak; lo_amp[, j] <- ap2$amp; lo_p[, j] <- r$p
  }
  tab[, `:=`(loto_phase_sd_h = circ_sd_h(lo_peak),
             loto_min_amp_ratio = apply(lo_amp, 1, min) / pmax(amp_log2, 1e-12),
             loto_max_p = apply(lo_p, 1, max),
             loto_worst_zt = zt[apply(lo_p, 1, which.max)])]

  # ---- edgeR quasi-likelihood (NB GLM) check
  y <- DGEList(cnt); y$samples$norm.factors <- res$norm_factors
  X <- cosinor_design(zt)
  y <- estimateDisp(y, X)
  qt <- glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)
  tab[, edgeR_p := qt$table$PValue]
  tab[, edgeR_q := p.adjust(edgeR_p, "BH")]

  tab[, `:=`(tissue = tissue, level = LEVEL, group = group, status = incl$status[i],
             n_cells_min = min(gc$n_cells), n_cells_total = sum(gc$n_cells), n_genes_tested = nrow(tab))]
  h <- rbindlist(hits)[, `:=`(tissue = tissue, level = LEVEL, group = group)]
  message(sprintf("  %-8s %-35s genes=%5d  q<0.1&FC>=1.5: %4d  null median %.0f", tissue, group, nrow(tab),
                  h$n_q10_fc15[1], median(h$n_q10_fc15[-1])))
  list(tab = tab, hits = h)
}

out <- bplapply(seq_len(nrow(incl)), run_group, BPPARAM = MulticoreParam(NCORES, stop.on.error = FALSE))
failed <- vapply(out, function(x) inherits(x, "error") || is.null(x$tab), TRUE)
if (any(failed)) message("FAILED groups: ", paste(incl$group[failed], collapse = "; "))
tabs <- rbindlist(lapply(out[!failed], `[[`, "tab"))
hits <- rbindlist(lapply(out[!failed], `[[`, "hits"))

setcolorder(tabs, c("tissue", "level", "group", "status", "gene", "n_cells_min", "n_cells_total", "n_genes_tested",
                    "AveExpr", "mesor", "a", "b", "se_a", "se_b", "df_total", "amp_log2", "ptt_fc", "peak_zt"))
for (t in unique(tabs$tissue)) {
  f <- file.path(out_dir, sprintf("rhythm_limma_%s_%s.csv", LEVEL, t))
  fwrite(tabs[tissue == t], f)
  message("[csv] ", f, " rows=", nrow(tabs[tissue == t]))
}
fwrite(hits, file.path(out_dir, sprintf("permutation_null_counts_%s.csv", LEVEL)))

# group-level global permutation test and empirical FDR at q<0.1 & FC>=1.5
glob <- hits[, .(obs_hits = n_q10_fc15[class == 1], null_mean = mean(n_q10_fc15[class != 1]),
                 null_median = median(n_q10_fc15[class != 1]), null_max = max(n_q10_fc15[class != 1]),
                 p_global = (1 + sum(n_q10_fc15[class != 1] >= n_q10_fc15[class == 1])) / .N,
                 null_mean_lowoverlap = mean(n_q10_fc15[class != 1 & overlap < 0.5])),
             by = .(tissue, level, group)]
glob[, empirical_fdr := pmin(1, null_mean / pmax(obs_hits, 1))]
fwrite(glob, file.path(out_dir, sprintf("rhythm_global_test_%s.csv", LEVEL)))
print(glob[order(tissue, -obs_hits)])
message("done")
