#!/usr/bin/env Rscript
# Module 2.3 (supplement) — likelihood-ratio rhythmicity test with finite-sample correction (diffCircadian
# LR_rhythmicity, Ding 2021) on TMM log2-CPM of each pseudobulk series; it was not installed when 32/33 ran, so the
# tier definitions (33) use edgeR-QL, RAIN and JTK as independent methods. This script adds the LR p-values and
# reports how many Tier-1 calls would change if LR counted as a further independent method.
# Outputs (results/32_rhythm_alt/): rhythm_lr_cell_type_<tissue>.csv, lr_tier_impact.csv
suppressPackageStartupMessages({ library(edgeR); library(data.table); library(diffCircadian); library(parallel) })
source("scripts/lib/cosinor.R")
NCORES <- as.integer(Sys.getenv("NCORES", "6"))
out_dir <- file.path(RESULTS, "32_rhythm_alt"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status != "excluded"]
cfg <- CFG$rhythm
impact <- list()
for (tis in c("cochlea", "bm")) {
  pb <- read_pseudobulk(tis)
  res <- rbindlist(mclapply(incl[tissue == tis, group], function(g) {
    gc <- group_counts(pb, design_dt, tis, g)
    cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
    lc <- cpm(calcNormFactors(DGEList(cnt)), log = TRUE, prior.count = 2)
    r <- t(apply(lc, 1, function(y) {
      o <- tryCatch(LR_rhythmicity(gc$zt, y, period = 24, method = "LR", FN = TRUE), error = function(e) NULL)
      if (is.null(o)) c(NA, NA, NA) else c(o$pvalue, o$amp, o$peakTime)
    }))
    data.table(tissue = tis, group = g, gene = rownames(lc), lr_p = r[, 1], lr_amp_log2 = r[, 2],
               lr_peak_zt = r[, 3] %% 24)
  }, mc.cores = NCORES))
  res[, lr_q := p.adjust(lr_p, "BH"), by = group]
  fwrite(res, file.path(out_dir, sprintf("rhythm_lr_cell_type_%s.csv", tis)))
  t <- fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", tis)))
  t[, lr_p := NULL]
  t <- merge(t, res[, .(group, gene, lr_p)], by = c("group", "gene"), all.x = TRUE)
  other <- t$q_BH < cfg$q_tier1 & t$q_perm < cfg$perm_fdr_tier1 & t$ptt_fc >= cfg$fc_threshold &
    t$loto_phase_sd_h < cfg$loto_phase_sd_h & !t$single_sample_driven
  with_lr <- other & (t$n_independent_p05 + (!is.na(t$lr_p) & t$lr_p < 0.05)) >= 1
  other[is.na(other)] <- FALSE; with_lr[is.na(with_lr)] <- FALSE
  t[, `:=`(other_ok = other, with_lr = with_lr)]
  impact[[tis]] <- t[, .(n_tier1 = sum(tier == "tier1"), n_tier1_recomputed = sum(other_ok & n_independent_p05 >= 1),
                         n_tier1_with_lr = sum(with_lr),
                         n_gained = sum(with_lr & tier != "tier1"), n_lr_p05 = sum(lr_p < 0.05, na.rm = TRUE),
                         frac_tier1_lr_p05 = mean(lr_p[tier == "tier1"] < 0.05, na.rm = TRUE)),
                     by = .(tissue, group)]
  message(tis, ": LR done for ", uniqueN(res$group), " cell types")
}
imp <- rbindlist(impact)
fwrite(imp, file.path(out_dir, "lr_tier_impact.csv"))
print(imp[, .(n_tier1 = sum(n_tier1), with_lr = sum(n_tier1_with_lr), gained = sum(n_gained),
              frac_tier1_lr_p05 = mean(frac_tier1_lr_p05, na.rm = TRUE)), by = tissue])
message("done")
