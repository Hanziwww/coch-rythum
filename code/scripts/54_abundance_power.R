#!/usr/bin/env Rscript
# Module 4 — power of abundance-rhythm tests per tissue × cell type (cf. 37_power.R for genes).
# (a) propeller (logit cell-type proportions, limma robust eBayes; as in 51): posterior residual variance s2.post and
#     df.total per cell type → power of the moderated 2-df F test, λ = 3A²/σ² with A = ln(FC)/2 for a peak-to-trough
#     odds ratio FC (Σcos² = Σsin² = 3 for 6 equally spaced samples).
# (b) unmoderated CLR cosinor (frame A_clr_all of 50_abundance; F(2, 3)) with each series' residual variance.
# Power at α = 0.05 and 0.01 for FC 1.25, 1.5, 2, 3, and the minimal FC detectable with 80 % power at α = 0.05.
# Outputs (results/54_abundance_power/): abundance_power.csv
suppressPackageStartupMessages({ library(limma); library(speckle); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "54_abundance_power"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ct <- fread(file.path(RESULTS, "00_audit/celltype_counts_by_sample.csv"))

power_F <- function(fc, s2, df2, alpha) {
  lambda <- 3 * (log(fc) / 2)^2 / s2
  pf(qf(1 - alpha, 2, df2), 2, df2, ncp = lambda, lower.tail = FALSE)
}
min_fc <- function(s2, df2, alpha = 0.05, target = 0.8) {
  f <- function(fc) power_F(fc, s2, df2, alpha) - target
  if (!is.finite(s2) || f(1000) < 0) return(NA_real_)
  uniroot(f, c(1.0001, 1000))$root
}
grid <- CJ(fc = c(1.25, 1.5, 2, 3), alpha = c(0.05, 0.01))
rows <- list()
for (tis in c("cochlea", "bm")) {
  d <- ct[tissue == tis]
  keep <- d[, .(tot = sum(n_cells)), by = cell_type][tot >= 60, cell_type]
  d <- d[cell_type %in% keep]
  cells <- d[, .(cluster = rep(cell_type, n_cells), sample = rep(sample_id, n_cells)), by = seq_len(nrow(d))]
  tp <- getTransformedProps(clusters = cells$cluster, sample = cells$sample, transform = "logit")
  smp <- unique(d[, .(sample_id, zt)])[match(colnames(tp$TransformedProps), sample_id)]
  fit <- eBayes(lmFit(tp$TransformedProps, cosinor_design(smp$zt)), robust = TRUE)
  for (i in seq_len(nrow(fit))) {
    s2 <- fit$s2.post[i]; df2 <- fit$df.total[i]
    rows[[length(rows) + 1]] <- grid[, .(tissue = tis, cell_type = rownames(fit)[i], method = "propeller_moderated",
      mean_proportion = mean(tp$Proportions[i, ]), s2 = s2, df_resid = df2, ptt_fc = fc, alpha = alpha,
      power = power_F(fc, s2, df2, alpha), min_fc_80pct_alpha05 = min_fc(s2, df2))]
  }
}
fr <- fread(file.path(RESULTS, "50_abundance/composition_frames.csv"))[frame == "A_clr_all"]
for (key in unique(fr[, paste(tissue, feature, sep = "||")])) {
  g <- fr[paste(tissue, feature, sep = "||") == key][order(zt)]
  if (nrow(g) != 6) next
  f <- lm(g$value_ln ~ cos(W24 * g$zt) + sin(W24 * g$zt))
  s2 <- sum(residuals(f)^2) / 3
  rows[[length(rows) + 1]] <- grid[, .(tissue = g$tissue[1], cell_type = g$feature[1], method = "clr_unmoderated",
    mean_proportion = NA_real_, s2 = s2, df_resid = 3, ptt_fc = fc, alpha = alpha, power = power_F(fc, s2, 3, alpha),
    min_fc_80pct_alpha05 = min_fc(s2, 3))]
}
pw <- rbindlist(rows)
fwrite(pw, file.path(out_dir, "abundance_power.csv"))
print(pw[alpha == 0.05 & ptt_fc == 2, .(tissue, cell_type, method, power = round(power, 2),
                                        min_fc = round(min_fc_80pct_alpha05, 2))][order(method, tissue, -power)])
message("done")
