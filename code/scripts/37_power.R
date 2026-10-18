#!/usr/bin/env Rscript
# Module 2.10 — power of the primary moderated cosinor F test per tissue × cell type × expression bin.
# A rhythm of log2 amplitude A gives non-centrality λ = A² / v for the 2-df F test, where v is the sampling variance of
# each cosinor coefficient; the moderated test has denominator df = d0 + 3 and uses the posterior variance.
# v per bin = median over the genes of that AveExpr quintile of s2.post × stdev.unscaled² (mean of the cos and sin terms).
# Round 9 correction: the first version used v = s2.post / 3 (Σcos² = 3 with unit weights), which ignores the voom
# precision weights — s2.post is the residual variance on the weighted scale (≈ 1 by construction), so power was
# understated by the mean weight. The weighted-scale s2 is kept as s2_median for transparency.
# Power is reported at nominal α = 0.05, 0.01 and 0.001 (≈ BH q < 0.1 when few genes are rhythmic) for
# peak-to-trough fold changes 1.25, 1.5, 2, 3, plus the minimum FC detectable with 80 % power at α = 0.001.
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "37_power"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status != "excluded"]
PB <- list(cochlea = read_pseudobulk("cochlea"), bm = read_pseudobulk("bm"))

power_F <- function(fc, v, df2, alpha) {
  A <- log2(fc) / 2
  lambda <- A^2 / v
  crit <- qf(1 - alpha, 2, df2)
  pf(crit, 2, df2, ncp = lambda, lower.tail = FALSE)
}
min_fc <- function(v, df2, alpha, target = 0.8) {
  f <- function(fc) power_F(fc, v, df2, alpha) - target
  if (f(64) < 0) return(NA_real_)
  uniroot(f, c(1.0001, 64))$root
}

rows <- list()
for (i in seq_len(nrow(incl))) {
  tissue <- incl$tissue[i]; group <- incl$group[i]
  gc <- group_counts(PB[[tissue]], design_dt, tissue, group)
  cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
  fit <- limma_cosinor(cnt, gc$zt)$fit
  bins <- cut(fit$Amean, quantile(fit$Amean, 0:5 / 5), include.lowest = TRUE, labels = paste0("Q", 1:5))
  for (b in levels(bins)) {
    s2 <- median(fit$s2.post[bins == b])
    v_all <- fit$s2.post * (fit$stdev.unscaled[, 2]^2 + fit$stdev.unscaled[, 3]^2) / 2
    v <- median(v_all[bins == b])
    df2 <- median(fit$df.total[bins == b])          # robust eBayes: df.prior / df.total are per-gene vectors
    for (fc in c(1.25, 1.5, 2, 3)) for (alpha in c(0.05, 0.01, 0.001)) {
      rows[[length(rows) + 1]] <- data.table(tissue = tissue, cell_type = group, status = incl$status[i],
        n_cells_min = min(gc$n_cells), expr_bin = b, median_log2cpm = median(fit$Amean[bins == b]),
        n_genes_bin = sum(bins == b), s2_median = s2, coef_var_median = v, df_prior_median = median(fit$df.prior[bins == b]),
        df_total = df2, ptt_fc = fc, alpha = alpha, power = power_F(fc, v, df2, alpha),
        min_fc_80pct_alpha001 = min_fc(v, df2, 0.001))
    }
  }
}
pw <- rbindlist(rows)
fwrite(pw, file.path(out_dir, "power_by_celltype.csv"))
print(pw[alpha == 0.001 & ptt_fc == 1.5, .(tissue, cell_type, expr_bin, power = round(power, 2),
                                            min_fc = round(min_fc_80pct_alpha001, 2))][expr_bin == "Q3"])
message("done")
