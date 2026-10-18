#!/usr/bin/env Rscript
# Round 2, WP4 — coverage of the delta-method 95 % phase CI on the data's own noise.
# As in 33b_simulation.R: 4 cell types per tissue, time labels relabelled by a random non-identity dihedral class,
# 24-h rhythms injected into 5 % of genes (FC 1.5, 2, 3; random phase) by binomial thinning; limma-voom cosinor refit.
# For injected genes: whether the 95 % delta-method CI (t, df.total; lib/cosinor.R phase_ci_delta) covers the true
# phase — overall, among genes with limma p < 0.05, and among phase-determinate estimates (half-width <= 3 h);
# median half-width. Output (results/r2_phase/): phase_ci_coverage.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table); library(parallel) })
source("scripts/lib/cosinor.R")
NC <- as.integer(Sys.getenv("NCORES", "4"))
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status == "primary"]
classes <- dihedral_classes(6)
jobs <- list()
for (tis in c("bm", "cochlea")) {
  pb <- read_pseudobulk(tis)
  groups <- incl[tissue == tis, group]
  libs <- sapply(groups, function(g) sum(group_counts(pb, design_dt, tis, g)$counts))
  pick <- names(sort(libs))[unique(round(quantile(seq_along(libs), c(0.1, 0.4, 0.7, 1))))]
  for (g in pick) {
    gc <- group_counts(pb, design_dt, tis, g)
    cnt0 <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
    for (rep in 1:3) for (fc in c(1.5, 2, 3)) {
      set.seed(1000 * rep + round(10 * fc) + match(g, groups))
      k <- sample(2:length(classes), 1)
      jobs[[length(jobs) + 1]] <- list(tissue = tis, cell_type = g, cnt0 = cnt0, zt = gc$zt[classes[[k]] + 1], fc = fc,
                                       replicate = rep, relabel_class = paste(classes[[k]], collapse = ""),
                                       seed = 7 * rep + round(10 * fc) + 100 * match(g, groups))
    }
  }
}
one <- function(j) {
  set.seed(j$seed)
  cnt <- j$cnt0; zt <- j$zt
  inj <- sample(nrow(cnt), round(0.05 * nrow(cnt)))
  phi <- runif(length(inj), 0, 24)
  A <- log2(j$fc) / 2
  f <- 2^(A * cos(W24 * outer(phi, zt, function(p, t) t - p)))
  f <- f / apply(f, 1, max)
  cnt[inj, ] <- matrix(rbinom(length(f), as.vector(cnt[inj, ]), as.vector(f)), nrow = length(inj))
  r <- limma_cosinor(cnt, zt)$table[inj]
  ci <- phase_ci_delta(r$a, r$b, r$se_a^2, r$se_b^2, 0, df = r$df_total)
  cover <- abs(circ_diff(ci$peak, phi)) <= ci$half_width_h
  det <- ci$half_width_h <= 3
  sig <- r$p < 0.05
  data.table(tissue = j$tissue, cell_type = j$cell_type, replicate = j$replicate, relabel_class = j$relabel_class,
             ptt_fc_injected = j$fc, n_injected = length(inj),
             coverage_all = mean(cover), coverage_p05 = if (any(sig)) mean(cover[sig]) else NA_real_,
             coverage_determinate = if (any(det)) mean(cover[det]) else NA_real_, frac_determinate = mean(det),
             median_halfwidth_h = median(ci$half_width_h), median_abs_error_h = median(abs(circ_diff(ci$peak, phi))))
}
res <- rbindlist(mclapply(jobs, one, mc.cores = NC))
out_dir <- file.path(RESULTS, "r2_phase"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(res, file.path(out_dir, "phase_ci_coverage.csv"))
print(res[, lapply(.SD, function(x) round(mean(x, na.rm = TRUE), 3)), by = .(tissue, ptt_fc_injected),
          .SDcols = c("coverage_all", "coverage_p05", "coverage_determinate", "frac_determinate", "median_halfwidth_h")])
message("done")
