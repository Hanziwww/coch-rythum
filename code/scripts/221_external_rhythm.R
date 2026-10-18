#!/usr/bin/env Rscript
# Round 2, WP3/WP4 — rhythm analysis of the additional external datasets (external/prepared_r2, from
# 220_external_prepare.py), with the round-1 model (71_external_rhythm.R): per dataset × tissue × condition,
# normcounts → log2(x + 1) and limma-trend, log2 arrays → limma; ~ cos + sin (24 h), robust eBayes, moderated F;
# mesor, amplitude, peak in the dataset's own time scale (ZT or CT). Where >= 10 samples allow it, a 2-harmonic model
# adds cos/sin(2ωt): moderated F of the 12-h terms given the 24-h terms (p_12h, amp_12h). Treatment effects on
# rhythms: dryR (drylm) for each non-control condition against the control of the same tissue.
# Outputs (results/r2_external/): external_rhythm_<dataset>.csv, external_dryR_<dataset>_<condition>.csv,
#   external_manifest_r2.csv
suppressPackageStartupMessages({ library(limma); library(data.table); library(dryR) })
source("scripts/lib/cosinor.R")
NC <- as.integer(Sys.getenv("NCORES", "4"))
out_dir <- file.path(RESULTS, "r2_external"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
prep <- file.path(ROOT, "external/prepared_r2")
datasets <- sub("__meta.csv$", "", list.files(prep, pattern = "__meta.csv$"))
man <- list()
for (ds in datasets) {
  meta <- fread(file.path(prep, paste0(ds, "__meta.csv")))
  ex <- fread(file.path(prep, paste0(ds, "__expr.csv")))
  genes <- ex[[1]]; ex <- as.matrix(ex[, -1]); rownames(ex) <- genes
  ex <- ex[, meta$sample, drop = FALSE]
  vt <- unique(meta$value_type)
  lx_all <- if (vt == "normcounts") log2(ex + 1) else ex
  res <- list()
  for (key in unique(meta[, paste(tissue, condition, sep = "|")])) {
    m <- meta[paste(tissue, condition, sep = "|") == key]
    x <- lx_all[, m$sample, drop = FALSE]
    x <- x[rowMeans(x) > quantile(rowMeans(x), 0.25, na.rm = TRUE) & apply(x, 1, sd) > 0, ]
    X <- cosinor_design(m$time_h)
    fit <- eBayes(lmFit(x, X), robust = TRUE, trend = TRUE)
    tt <- topTable(fit, coef = 2:3, number = Inf, sort.by = "none")
    ap <- amp_phase(fit$coefficients[, 2], fit$coefficients[, 3])
    r <- data.table(dataset = ds, tissue = m$tissue[1], condition = m$condition[1], time_type = m$time_type[1],
                    gene = rownames(fit$coefficients), n_samples = ncol(x), mesor = fit$coefficients[, 1],
                    a = fit$coefficients[, 2], b = fit$coefficients[, 3],
                    se_a = fit$stdev.unscaled[, 2] * sqrt(fit$s2.post), se_b = fit$stdev.unscaled[, 3] * sqrt(fit$s2.post),
                    df_total = fit$df.total, amp_log2 = ap$amp, ptt_fc = 2^(2 * ap$amp), peak_h = ap$peak,
                    F = tt$F, p = tt$P.Value, q_BH = tt$adj.P.Val)
    if (ncol(x) >= 10 && uniqueN(m$time_h) >= 5) {
      X2 <- cbind(X, cos2 = cos(2 * W24 * m$time_h), sin2 = sin(2 * W24 * m$time_h))
      f2 <- eBayes(lmFit(x, X2), robust = TRUE, trend = TRUE)
      t2 <- topTable(f2, coef = 4:5, number = Inf, sort.by = "none")
      r[, `:=`(amp_12h = sqrt(f2$coefficients[, 4]^2 + f2$coefficients[, 5]^2), p_12h = t2$P.Value,
               q_12h = t2$adj.P.Val)]
    }
    res[[key]] <- r
    message(sprintf("%s %s: %d samples, %d genes, q<0.05: %d", ds, key, ncol(x), nrow(r), sum(r$q_BH < 0.05)))
  }
  fwrite(rbindlist(res, fill = TRUE), file.path(out_dir, sprintf("external_rhythm_%s.csv", ds)))
  for (tis in unique(meta$tissue)) {
    mt <- meta[tissue == tis]
    if (!"control" %in% mt$condition) next
    for (cond in setdiff(unique(mt$condition), "control")) {
      m <- mt[condition %in% c("control", cond)]
      x <- lx_all[, m$sample, drop = FALSE]
      x <- x[rowMeans(x) > quantile(rowMeans(x), 0.25, na.rm = TRUE) & apply(x, 1, sd) > 0, ]
      dr <- tryCatch(drylm(x, group = m$condition, time = m$time_h, period = 24, sample_name = m$sample, n.cores = NC),
                     error = function(e) { message("dryR failed ", ds, " ", cond, ": ", conditionMessage(e)); NULL })
      if (!is.null(dr)) {
        d <- as.data.table(dr$results, keep.rownames = "gene")
        d[, `:=`(dataset = ds, tissue = tis, contrast = paste0(cond, "_vs_control"))]
        fwrite(d, file.path(out_dir, sprintf("external_dryR_%s_%s.csv", ds, cond)))
      }
    }
  }
  meta[, dataset := ds]
  man[[ds]] <- meta[, .(n_samples = .N, time_points = paste(sort(unique(round(time_h, 1))), collapse = ","),
                        samples_per_time = round(.N / uniqueN(time_h), 1)),
                    by = .(dataset, tissue, condition, time_type, value_type)]
}
fwrite(rbindlist(man), file.path(out_dir, "external_manifest_r2.csv"))
message("done")
