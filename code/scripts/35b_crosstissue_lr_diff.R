#!/usr/bin/env Rscript
# Module 2.6 (supplement) — labelled dryR model classes and likelihood-ratio tests of differential amplitude and
# phase (diffCircadian::LR_diff, finite-sample corrected; Ding 2021) between the cochlea and temporal-BM cohorts.
# dryR model numbers are mapped to labels from which tissue's rhythm parameters are NA/zero in each model
# (arrhythmic / cochlea_only / bm_only / shared / different). For every shared cell type, genes with a rhythmic
# dryR model (BICW >= 0.5) are tested on within-tissue TMM log2-CPM; peak phases and amplitudes come from OLS cosinor
# fits of the same values. Δmesor (platform) is not tested.
# Outputs (results/35_crosstissue/): crosstissue_dryR_labelled_counts.csv, crosstissue_lr_diff.csv
suppressPackageStartupMessages({ library(edgeR); library(data.table); library(nloptr); library(diffCircadian)
  library(parallel) })
source("scripts/lib/cosinor.R")
NCORES <- as.integer(Sys.getenv("NCORES", "6"))
out_dir <- file.path(RESULTS, "35_crosstissue")
dry <- fread(file.path(out_dir, "crosstissue_dryR.csv"))
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
PB <- list(cochlea = read_pseudobulk("cochlea"), bm = read_pseudobulk("bm"))

amp_c <- grep("^amp_.*cochlea", names(dry), value = TRUE)[1]
amp_b <- grep("^amp_.*bm", names(dry), value = TRUE)[1]
lab <- dry[, .(z_c = mean(is.na(get(amp_c)) | abs(get(amp_c)) < 1e-12),
               z_b = mean(is.na(get(amp_b)) | abs(get(amp_b)) < 1e-12),
               same = mean(abs(get(amp_c) - get(amp_b)) < 1e-9, na.rm = TRUE)), by = chosen_model]
lab[, label := fifelse(z_c > 0.5 & z_b > 0.5, "arrhythmic", fifelse(z_c > 0.5, "bm_only",
                fifelse(z_b > 0.5, "cochlea_only", fifelse(same > 0.5, "shared", "different"))))]
message("dryR model labels: ", paste(lab$chosen_model, lab$label, sep = "=", collapse = ", "))
dry <- merge(dry, lab[, .(chosen_model, model_label = label)], by = "chosen_model")
cnt <- dcast(dry[, .N, by = .(cell_type, model_label)], cell_type ~ model_label, value.var = "N", fill = 0)
conf <- dcast(dry[chosen_model_BICW >= 0.5, .N, by = .(cell_type, model_label)], cell_type ~ model_label,
              value.var = "N", fill = 0)
setnames(conf, setdiff(names(conf), "cell_type"), paste0(setdiff(names(conf), "cell_type"), "_bicw05"))
fwrite(merge(cnt, conf, by = "cell_type", all = TRUE), file.path(out_dir, "crosstissue_dryR_labelled_counts.csv"))

cosfit <- function(y, t) {
  f <- lm(y ~ cos(W24 * t) + sin(W24 * t)); ap <- amp_phase(coef(f)[2], coef(f)[3]); c(ap$amp, ap$peak)
}
rows <- list()
for (ct in unique(dry$cell_type)) {
  cand <- dry[cell_type == ct & model_label != "arrhythmic" & chosen_model_BICW >= 0.5]
  if (!nrow(cand)) next
  gcc <- group_counts(PB$cochlea, design_dt, "cochlea", ct)
  gbm <- group_counts(PB$bm, design_dt, "bm", ct)
  keep <- expr_filter(gcc$counts) & expr_filter(gbm$counts)
  lc <- cpm(calcNormFactors(DGEList(gcc$counts[keep, ])), log = TRUE, prior.count = 1)
  lb <- cpm(calcNormFactors(DGEList(gbm$counts[keep, ])), log = TRUE, prior.count = 1)
  genes <- intersect(cand$gene, rownames(lc))
  res <- rbindlist(mclapply(genes, mc.cores = NCORES, FUN = function(g) {
    yc <- as.numeric(lc[g, ]); yb <- as.numeric(lb[g, ])
    pa <- tryCatch(LR_diff(gcc$zt, yc, gbm$zt, yb, period = 24, FN = TRUE, type = "amplitude")$pvalue,
                   error = function(e) NA_real_)
    pp <- tryCatch(LR_diff(gcc$zt, yc, gbm$zt, yb, period = 24, FN = TRUE, type = "phase")$pvalue,
                   error = function(e) NA_real_)
    fc <- cosfit(yc, gcc$zt); fb <- cosfit(yb, gbm$zt)
    data.table(cell_type = ct, gene = g, amp_log2_cochlea = fc[1], peak_zt_cochlea = fc[2],
               amp_log2_bm = fb[1], peak_zt_bm = fb[2], delta_peak_h_cochlea_minus_bm = circ_diff(fc[2], fb[2]),
               log2_amp_ratio_cochlea_bm = log2(fc[1] / fb[1]), p_lr_amplitude = pa, p_lr_phase = pp)
  }))
  res <- merge(res, cand[, .(gene, model_label, chosen_model_BICW)], by = "gene")
  rows[[ct]] <- res
  message(sprintf("  %-35s %4d genes tested", ct, nrow(res)))
}
out <- rbindlist(rows)
out[, `:=`(q_lr_amplitude = p.adjust(p_lr_amplitude, "BH"), q_lr_phase = p.adjust(p_lr_phase, "BH")), by = cell_type]
fwrite(out, file.path(out_dir, "crosstissue_lr_diff.csv"))
print(out[, .(n = .N, amp_q05 = sum(q_lr_amplitude < 0.05, na.rm = TRUE), phase_q05 = sum(q_lr_phase < 0.05, na.rm = TRUE)),
          by = model_label])
message("done")
