#!/usr/bin/env Rscript
# Module 5.4c — BayesPrism deconvolution of external circadian bulk data with our atlas as reference
# (BM reference for GSE198770; cochlea reference for GSE88954 when prepared). Reference: <= 400 cells per
# gold-standard cell type (75b_export_reference.py); genes shared with the bulk; outlier genes removed by
# BayesPrism defaults. Fractions per sample → cosinor per cell type × condition (replicates), and condition
# effects (mean shift and rhythm difference, F test) — e.g. neutrophil depletion must lower neutrophil fractions.
# Outputs (results/75_deconvolution/): bayesprism_fractions.csv, bayesprism_abundance_rhythm.csv,
#   bayesprism_condition_effects.csv
suppressPackageStartupMessages({ library(BayesPrism); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "75_deconvolution"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
prep <- file.path(ROOT, "external/prepared")
ref_dir <- file.path(CACHE, "75_bayesprism")
MATCH <- list(GSE198770 = "bm", GSE88954 = "cochlea")
fr_all <- list(); rh <- list(); ce <- list()
for (ds in intersect(names(MATCH), sub("__meta.csv$", "", list.files(prep, "__meta.csv$")))) {
  tis <- MATCH[[ds]]
  meta <- fread(file.path(prep, paste0(ds, "__meta.csv")))
  bk <- fread(file.path(prep, paste0(ds, "__expr.csv")))
  genes <- bk[[1]]; bk <- as.matrix(bk[, -1]); rownames(bk) <- genes
  bk <- t(round(bk[, meta$sample]))
  ref <- fread(file.path(ref_dir, sprintf("reference_counts_%s.csv.gz", tis)))
  cells <- ref[[1]]; ref <- as.matrix(ref[, -1]); rownames(ref) <- cells
  lab <- fread(file.path(ref_dir, sprintf("reference_labels_%s.csv", tis)))
  common <- intersect(colnames(ref), colnames(bk))
  message(ds, ": ", length(common), " shared genes, ", nrow(ref), " reference cells, ", nrow(bk), " bulk samples")
  prism <- new.prism(reference = ref[, common], mixture = bk[, common], input.type = "count.matrix",
                     cell.type.labels = lab$cell_type, cell.state.labels = lab$cell_type, key = NULL,
                     outlier.cut = 0.01, outlier.fraction = 0.1)
  res <- run.prism(prism = prism, n.cores = as.integer(Sys.getenv("NCORES", "8")))
  theta <- get.fraction(bp = res, which.theta = "final", state.or.type = "type")
  fr <- as.data.table(as.table(theta))
  setnames(fr, c("sample", "cell_type", "fraction"))
  fr <- merge(fr, meta[, .(sample, time_h, time_type, condition, tissue)], by = "sample")
  fr[, dataset := ds]
  fr_all[[ds]] <- fr
  for (key in unique(fr[, paste(condition, cell_type, sep = "||")])) {
    g <- fr[paste(condition, cell_type, sep = "||") == key]
    if (mean(g$fraction) < 0.002) next
    y <- qlogis(pmin(pmax(g$fraction, 1e-4), 1 - 1e-4))
    X <- cosinor_design(g$time_h)
    f <- lm(y ~ X[, 2] + X[, 3])
    a0 <- anova(lm(y ~ 1), f)
    ap <- amp_phase(coef(f)[2], coef(f)[3])
    rh[[paste(ds, key)]] <- data.table(dataset = ds, condition = g$condition[1], cell_type = g$cell_type[1],
                                       n = nrow(g), mean_fraction = mean(g$fraction), amp_logit = ap$amp,
                                       peak_h = ap$peak, F = a0$F[2], p = a0$`Pr(>F)`[2])
  }
  if (uniqueN(fr$condition) > 1) for (ct in unique(fr$cell_type)) {
    g <- fr[cell_type == ct]
    if (mean(g$fraction) < 0.002) next
    g[, `:=`(y = qlogis(pmin(pmax(fraction, 1e-4), 1 - 1e-4)), cv = cos(W24 * time_h), sv = sin(W24 * time_h))]
    full <- lm(y ~ condition * (cv + sv), g); red <- lm(y ~ condition + cv + sv, g); base <- lm(y ~ cv + sv, g)
    ce[[paste(ds, ct)]] <- data.table(dataset = ds, cell_type = ct, mean_by_condition = paste(g[, .(m = round(mean(fraction), 4)), by = condition][, paste(condition, m, sep = "=")], collapse = ";"),
                                      p_mean_shift = anova(base, red)$`Pr(>F)`[2], p_rhythm_difference = anova(red, full)$`Pr(>F)`[2])
  }
}
fwrite(rbindlist(fr_all), file.path(out_dir, "bayesprism_fractions.csv"))
fwrite(rbindlist(rh), file.path(out_dir, "bayesprism_abundance_rhythm.csv"))
fwrite(rbindlist(ce), file.path(out_dir, "bayesprism_condition_effects.csv"))
print(rbindlist(ce)); print(rbindlist(rh)[order(p)][1:15])
message("done")
