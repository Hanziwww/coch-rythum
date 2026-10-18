#!/usr/bin/env Rscript
# Module 2.3 (sensitivity) — cell-level negative-binomial GLMM with a sample random intercept (NEBULA-type model;
# NEBULA could not be installed, glmmTMB is used instead) for Tier-1/2 candidates and core-clock genes.
# count_ic ~ cos(ωt) + sin(ωt) + offset(log library size) + (1 | sample), family nbinom2; LR test vs the model
# without cos/sin (2 df). <= 300 cells per cell type × sample (random subsample, seed 1).
# Outputs (results/38_cell_level/): cell_level_glmm.csv
suppressPackageStartupMessages({ library(glmmTMB); library(data.table); library(BiocParallel) })
source("scripts/lib/cosinor.R")
NCORES <- as.integer(Sys.getenv("NCORES", "24"))
out_dir <- file.path(RESULTS, "38_cell_level"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
clock <- c("Npas2", "Clock", "Nr1d1", "Nr1d2", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Dbp", "Tef", "Hlf", "Nfil3",
           "Bhlhe40", "Bhlhe41", "Ciart")
inp <- file.path(CACHE, "38_cell_level")
rows <- list()
for (tis in c("cochlea", "bm")) {
  f <- file.path(inp, sprintf("cells_%s.csv.gz", tis))       # written by 38_export_cells.py
  if (!file.exists(f)) { message("missing ", f); next }
  d <- fread(f)
  tiers <- fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", tis)),
                 select = c("group", "gene", "tier", "p"))
  cand <- tiers[tier == "tier1" | gene %in% clock]
  cand <- cand[gene %in% names(d)]
  jobs <- split(cand, by = c("group", "gene"))
  res <- bplapply(jobs, function(j) {
    sub <- d[cell_type == j$group]
    if (nrow(sub) < 200) return(NULL)
    sub[, `:=`(y = get(j$gene), cosv = cos(W24 * zt), sinv = sin(W24 * zt))]
    m1 <- tryCatch(glmmTMB(y ~ cosv + sinv + offset(log(lib)) + (1 | sample_id), data = sub, family = nbinom2),
                   error = function(e) NULL)
    m0 <- tryCatch(glmmTMB(y ~ 1 + offset(log(lib)) + (1 | sample_id), data = sub, family = nbinom2),
                   error = function(e) NULL)
    if (is.null(m1) || is.null(m0)) return(NULL)
    lr <- 2 * (as.numeric(logLik(m1)) - as.numeric(logLik(m0)))
    b <- fixef(m1)$cond
    ap <- amp_phase(b["cosv"], b["sinv"])
    data.table(tissue = tis, cell_type = j$group, gene = j$gene, tier = j$tier, pseudobulk_p = j$p, n_cells = nrow(sub),
               amp_log2 = ap$amp / log(2), peak_zt = ap$peak, LR = lr, p_glmm = pchisq(lr, 2, lower.tail = FALSE),
               sample_sd = sqrt(as.numeric(VarCorr(m1)$cond$sample_id)))
  }, BPPARAM = MulticoreParam(NCORES))
  rows[[tis]] <- rbindlist(res)
}
out <- rbindlist(rows)
fwrite(out, file.path(out_dir, "cell_level_glmm.csv"))
print(out[, .(n = .N, glmm_p05 = sum(p_glmm < 0.05), frac = mean(p_glmm < 0.05)), by = .(tissue, tier)])
message("done")
