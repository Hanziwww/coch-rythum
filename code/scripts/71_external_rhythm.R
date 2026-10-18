#!/usr/bin/env Rscript
# Module 5.4a — harmonised rhythm analysis of external circadian bulk datasets (with biological replicates).
# Input (from 70_external_prepare.py): external/prepared/<dataset>__expr.csv (genes × samples; counts or log2 values)
#   and <dataset>__meta.csv (sample, time_h, time_type [ZT|CT], condition, tissue, replicate, value_type [counts|log2]).
# Per dataset × tissue × condition: counts → TMM + voom; log2 arrays → limma directly; ~ cos + sin, robust eBayes,
# moderated F; mesor/amplitude/peak (in the dataset's own time scale, CT or ZT). Condition comparisons
# (sham vs ADX, control vs neutrophil depletion) by dryR model selection (counts: dryseq; log2: drylm).
# Outputs (results/71_external/): external_rhythm_<dataset>.csv, external_dryR_<dataset>.csv, external_manifest.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table); library(dryR) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "71_external"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
prep <- file.path(ROOT, "external/prepared")
datasets <- sub("__meta.csv$", "", list.files(prep, pattern = "__meta.csv$"))
man <- list()
for (ds in datasets) {
  meta <- fread(file.path(prep, paste0(ds, "__meta.csv")))
  ex <- fread(file.path(prep, paste0(ds, "__expr.csv")))
  genes <- ex[[1]]; ex <- as.matrix(ex[, -1]); rownames(ex) <- genes
  ex <- ex[, meta$sample, drop = FALSE]
  vt <- unique(meta$value_type)
  res <- list()
  for (key in unique(meta[, paste(tissue, condition, sep = "|")])) {
    m <- meta[paste(tissue, condition, sep = "|") == key]
    x <- ex[, m$sample, drop = FALSE]
    X <- cosinor_design(m$time_h)
    if (vt == "counts") {
      keep <- expr_filter(x, min_samples = max(3, floor(ncol(x) / 3)))
      y <- calcNormFactors(DGEList(round(x[keep, ])))
      v <- voom(y, X)
    } else {
      lx <- if (vt == "normcounts") log2(x + 1) else x
      v <- lx[rowMeans(lx) > quantile(rowMeans(lx), 0.25, na.rm = TRUE), ]
    }
    fit <- eBayes(lmFit(v, X), robust = TRUE, trend = vt != "counts")
    tt <- topTable(fit, coef = 2:3, number = Inf, sort.by = "none")
    ap <- amp_phase(fit$coefficients[, 2], fit$coefficients[, 3])
    res[[key]] <- data.table(dataset = ds, tissue = m$tissue[1], condition = m$condition[1],
                             time_type = m$time_type[1], gene = rownames(fit$coefficients),
                             n_samples = ncol(x), mesor = fit$coefficients[, 1], a = fit$coefficients[, 2],
                             b = fit$coefficients[, 3], se_a = fit$stdev.unscaled[, 2] * sqrt(fit$s2.post),
                             se_b = fit$stdev.unscaled[, 3] * sqrt(fit$s2.post), amp_log2 = ap$amp,
                             ptt_fc = 2^(2 * ap$amp), peak_h = ap$peak, F = tt$F, p = tt$P.Value, q_BH = tt$adj.P.Val)
    message(sprintf("%s %s: %d samples, %d genes, q<0.05: %d", ds, key, ncol(x), nrow(tt), sum(tt$adj.P.Val < 0.05)))
  }
  rr <- rbindlist(res)
  fwrite(rr, file.path(out_dir, sprintf("external_rhythm_%s.csv", ds)))
  # condition comparison with dryR (only if >= 2 conditions within a tissue)
  for (tis in unique(meta$tissue)) {
    m <- meta[tissue == tis]
    if (length(unique(m$condition)) < 2) next
    x <- ex[, m$sample, drop = FALSE]
    dr <- tryCatch({
      if (vt == "counts") {
        keep <- expr_filter(x, min_samples = max(3, floor(ncol(x) / 3)))
        xi <- round(x[keep, ]); storage.mode(xi) <- "integer"
        dryseq(xi, group = m$condition, time = m$time_h, period = 24, sample_name = m$sample)
      } else {
        lx <- if (vt == "normcounts") log2(x + 1) else x
        lx <- lx[rowMeans(lx) > quantile(rowMeans(lx), 0.25, na.rm = TRUE), ]
        drylm(lx, group = m$condition, time = m$time_h, period = 24, sample_name = m$sample, n.cores = 8)
      }
    }, error = function(e) { message("dryR failed ", ds, " ", tis, ": ", conditionMessage(e)); NULL })
    if (!is.null(dr)) {
      d <- as.data.table(dr$results, keep.rownames = "gene")
      d[, `:=`(dataset = ds, tissue = tis)]
      fwrite(d, file.path(out_dir, sprintf("external_dryR_%s_%s.csv", ds, tis)))
    }
  }
  meta[, dataset := ds]
  man[[ds]] <- meta[, .(n_samples = .N, time_points = paste(sort(unique(time_h)), collapse = ","),
                        replicates_per_time = round(.N / uniqueN(time_h), 1)),
                    by = .(dataset, tissue, condition, time_type, value_type)]
}
fwrite(rbindlist(man), file.path(out_dir, "external_manifest.csv"))
message("done")
