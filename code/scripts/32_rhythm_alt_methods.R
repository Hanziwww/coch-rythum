#!/usr/bin/env Rscript
# Module 2.3 — independent rhythm methods on the same pseudobulk data as 31_rhythm_limma.R.
#   RAIN (umbrella/non-sinusoidal, independent samples), JTK_CYCLE (MetaCycle), LR_rhythmicity (diffCircadian,
#   if installed), and LimoRhyde2 (cosinor basis + mash shrinkage across genes → posterior peak-to-trough
#   amplitude, phase and 90 % credible intervals).
# Inputs: log2-CPM from TMM-normalised pseudobulk (same gene filter as 31).
# Usage: Rscript scripts/32_rhythm_alt_methods.R [cell_type|cell_class]
suppressPackageStartupMessages({
  library(limma); library(edgeR); library(data.table); library(BiocParallel)
  library(rain); library(MetaCycle); library(limorhyde2)
})
source("scripts/lib/cosinor.R")
HAS_LR <- requireNamespace("diffCircadian", quietly = TRUE)

args <- commandArgs(trailingOnly = TRUE)
LEVEL <- if (length(args)) args[1] else "cell_type"
NCORES <- as.integer(Sys.getenv("NCORES", "32"))
out_dir <- file.path(RESULTS, "32_rhythm_alt"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tmp_root <- file.path(CACHE, "32_metacycle"); dir.create(tmp_root, recursive = TRUE, showWarnings = FALSE)

design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == LEVEL & status != "excluded"]
PB <- list(cochlea = read_pseudobulk("cochlea", LEVEL), bm = read_pseudobulk("bm", LEVEL))
message(sprintf("%s: %d groups; diffCircadian %s", LEVEL, nrow(incl), if (HAS_LR) "available" else "NOT installed"))

run_group <- function(i) {
  tissue <- incl$tissue[i]; group <- incl$group[i]
  gc <- group_counts(PB[[tissue]], design_dt, tissue, group, LEVEL)
  cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
  zt <- gc$zt
  y <- calcNormFactors(DGEList(cnt))
  lcpm <- cpm(y, log = TRUE, prior.count = 1)
  out <- data.table(gene = rownames(lcpm))

  # RAIN: one sample per 4-h step over one 24-h period, samples independent
  rr <- rain(t(lcpm), deltat = 4, period = 24, measure.sequence = rep(1, 6), method = "independent",
             verbose = FALSE)
  out[, `:=`(rain_p = rr$pVal, rain_peak_zt = (zt[1] + rr$phase) %% 24, rain_peak_shape = rr$peak.shape)]

  # JTK_CYCLE via MetaCycle (file based)
  d <- file.path(tmp_root, sprintf("%s__%s", tissue, gsub("[^A-Za-z0-9]+", "_", group)))
  dir.create(d, showWarnings = FALSE)
  infile <- file.path(d, "in.csv")
  fwrite(data.table(CycID = rownames(lcpm), lcpm), infile)
  invisible(capture.output(meta2d(infile = infile, filestyle = "csv", outdir = d, timepoints = zt,
                                  cycMethod = c("JTK"), minper = 24, maxper = 24, outIntegration = "noIntegration",
                                  parallelize = FALSE)))
  jtk <- fread(list.files(d, pattern = "^JTKresult", full.names = TRUE)[1])
  out <- merge(out, jtk[, .(gene = CycID, jtk_p = ADJ.P, jtk_bh = BH.Q, jtk_lag = LAG, jtk_amp = AMP)],
               by = "gene", all.x = TRUE)
  out[, jtk_peak_zt := (zt[1] + jtk_lag) %% 24]

  # likelihood-ratio rhythmicity test with finite-sample correction (Ding et al. 2021)
  if (HAS_LR) {
    lr <- vapply(seq_len(nrow(lcpm)), function(k)
      tryCatch(diffCircadian::LR_rhythmicity(zt, lcpm[k, ], period = 24, FN = TRUE)$pvalue,
               error = function(e) NA_real_), numeric(1))
    out[match(rownames(lcpm), gene), lr_p := lr]
  }

  # LimoRhyde2 (cosinor basis, voom, mash shrinkage across genes). With sinusoid = TRUE the two basis coefficients
  # are the cos/sin terms, so posterior amplitude/phase and 90 % credible intervals follow directly from mash's
  # posterior means/SDs (vectorised draws) — equivalent to getStatsIntervals but ~100× faster.
  meta <- data.table(sample = colnames(cnt), time = zt)
  l2 <- tryCatch({
    fit <- getPosteriorFit(getModelFit(cnt, meta, period = 24, sinusoid = TRUE, timeColname = "time",
                                       method = "voom"))
    pm <- fit$mashCoefficients
    ps <- fit$mashFits[[1]]$result$PosteriorSD
    lf <- fit$mashFits[[1]]$result$lfsr
    a <- pm[, "basis1_shift1"]; b <- pm[, "basis2_shift1"]
    st_ref <- as.data.table(getRhythmStats(fit, fitType = "posterior_mean", features = rownames(pm)[1:20]))
    ap <- amp_phase(a, b)
    d <- circ_diff(ap$peak[1:20], st_ref$peak_phase)
    if (median(abs(d)) > 0.1) {                       # basis order differs: basis1 = sin, basis2 = cos
      tmp <- a; a <- b; b <- tmp; ap <- amp_phase(a, b)
      d <- circ_diff(ap$peak[1:20], st_ref$peak_phase)
    }
    stopifnot(median(abs(d)) < 0.1)
    G <- length(a); nd <- 500
    A <- a + ps[, "basis1_shift1"] * matrix(rnorm(G * nd), G)
    B <- b + ps[, "basis2_shift1"] * matrix(rnorm(G * nd), G)
    amp <- 2 * sqrt(A^2 + B^2)
    pk <- (atan2(B, A) %% (2 * pi)) * 12 / pi
    dev <- circ_diff(pk, ap$peak)
    qs <- function(m, p) matrixStats::rowQuantiles(m, probs = p)
    data.table(gene = rownames(pm), l2_ptt_amp = 2 * ap$amp, l2_peak_zt = ap$peak, l2_mean = pm[, 1],
               l2_ptt_amp_lo = qs(amp, 0.05), l2_ptt_amp_hi = qs(amp, 0.95),
               l2_peak_lo = (ap$peak + qs(dev, 0.05)) %% 24, l2_peak_hi = (ap$peak + qs(dev, 0.95)) %% 24,
               l2_lfsr_min = pmin(lf[, 1], lf[, 2]))
  }, error = function(e) { message("limorhyde2 failed for ", group, ": ", conditionMessage(e)); NULL })
  if (!is.null(l2)) out <- merge(out, l2, by = "gene", all.x = TRUE)

  out[, `:=`(tissue = tissue, level = LEVEL, group = group)]
  message(sprintf("  %-8s %-35s genes=%5d  RAIN p<0.01: %4d  JTK p<0.01: %4d", tissue, group, nrow(out),
                  sum(out$rain_p < 0.01, na.rm = TRUE), sum(out$jtk_p < 0.01, na.rm = TRUE)))
  out
}

res <- bplapply(seq_len(nrow(incl)), run_group, BPPARAM = MulticoreParam(NCORES, stop.on.error = FALSE))
failed <- vapply(res, function(x) inherits(x, "error"), TRUE)
if (any(failed)) message("FAILED groups: ", paste(incl$group[failed], collapse = "; "),
                         "\n", paste(vapply(res[failed], conditionMessage, ""), collapse = "\n"))
tab <- rbindlist(res[!failed], fill = TRUE)
for (t in unique(tab$tissue)) {
  f <- file.path(out_dir, sprintf("rhythm_alt_%s_%s.csv", LEVEL, t))
  fwrite(tab[tissue == t], f)
  message("[csv] ", f, " rows=", nrow(tab[tissue == t]))
}
message("done")
