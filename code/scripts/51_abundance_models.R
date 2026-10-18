#!/usr/bin/env Rscript
# Module 4 — count-based compositional rhythm models per tissue.
#  (a) propeller-style: logit-transformed cell-type proportions (speckle::getTransformedProps) + limma
#      ~ cos + sin with robust empirical Bayes across cell types (moderated F, 2 df).
#  (b) sccomp (if installed): Bayesian beta-binomial with outlier detection, formula ~ cos + sin;
#      cos/sin effects with credible intervals and FDR; amplitude/phase from posterior-mean effects.
# Outputs (results/51_abundance_models/): abundance_rhythm_propeller.csv, abundance_rhythm_sccomp.csv
suppressPackageStartupMessages({ library(limma); library(speckle); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "51_abundance_models"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ct <- fread(file.path(RESULTS, "00_audit/celltype_counts_by_sample.csv"))

prop_rows <- list()
for (tis in c("cochlea", "bm")) {
  d <- ct[tissue == tis]
  keep <- d[, .(tot = sum(n_cells), mn = min(n_cells)), by = cell_type][tot >= 60, cell_type]
  d <- d[cell_type %in% keep]
  cells <- d[, .(cluster = rep(cell_type, n_cells), sample = rep(sample_id, n_cells)), by = seq_len(nrow(d))]
  tp <- getTransformedProps(clusters = cells$cluster, sample = cells$sample, transform = "logit")
  smp <- unique(d[, .(sample_id, zt)])[match(colnames(tp$TransformedProps), sample_id)]
  X <- cosinor_design(smp$zt)
  fit <- eBayes(lmFit(tp$TransformedProps, X), robust = TRUE)
  tt <- topTable(fit, coef = 2:3, number = Inf, sort.by = "none")
  ap <- amp_phase(fit$coefficients[, 2], fit$coefficients[, 3])
  prop_rows[[tis]] <- data.table(tissue = tis, cell_type = rownames(fit$coefficients),
                                 mean_proportion = rowMeans(tp$Proportions),
                                 mesor_logit = fit$coefficients[, 1], a_cos = fit$coefficients[, 2],
                                 b_sin = fit$coefficients[, 3], amp_logit = ap$amp, peak_zt = ap$peak,
                                 F = tt$F, p = tt$P.Value, q_BH = tt$adj.P.Val)
}
prop <- rbindlist(prop_rows)
fwrite(prop, file.path(out_dir, "abundance_rhythm_propeller.csv"))
print(prop[order(tissue, p)][, head(.SD, 8), by = tissue])

if (requireNamespace("sccomp", quietly = TRUE)) {
  suppressPackageStartupMessages(library(sccomp))
  cmdstanr::set_cmdstan_path(file.path(Sys.getenv("COCH_ENVS", "envs"), "rcirc/bin/cmdstan"))
  stan_cache <- file.path(CACHE, "sccomp_models"); dir.create(stan_cache, recursive = TRUE, showWarnings = FALSE)
  sc_rows <- list()
  for (tis in c("cochlea", "bm")) {
    d <- ct[tissue == tis][, .(sample_id, cell_type, n_cells, zt)]
    keep <- d[, .(tot = sum(n_cells)), by = cell_type][tot >= 60, cell_type]
    d <- d[cell_type %in% keep]
    d[, `:=`(cosv = cos(W24 * zt), sinv = sin(W24 * zt))]
    est <- tryCatch({
      sccomp_estimate(as.data.frame(d), formula_composition = ~ cosv + sinv, .sample = sample_id,
                      .cell_group = cell_type, .count = n_cells, cores = 8, verbose = FALSE,
                      cache_stan_model = stan_cache) |>
        sccomp_remove_outliers(cores = 8, verbose = FALSE, cache_stan_model = stan_cache) |>
        sccomp_test()
    }, error = function(e) { message("sccomp failed for ", tis, ": ", conditionMessage(e)); NULL })
    if (is.null(est)) next
    e <- as.data.table(est)
    w <- dcast(e[parameter %in% c("cosv", "sinv")], cell_type ~ parameter,
               value.var = c("c_effect", "c_lower", "c_upper", "c_FDR"))
    ap <- amp_phase(w$c_effect_cosv, w$c_effect_sinv)
    w[, `:=`(tissue = tis, amp_logit = ap$amp, peak_zt = ap$peak,
             min_FDR = pmin(c_FDR_cosv, c_FDR_sinv))]
    sc_rows[[tis]] <- w
  }
  if (length(sc_rows)) {
    sc <- rbindlist(sc_rows, fill = TRUE)
    fwrite(sc, file.path(out_dir, "abundance_rhythm_sccomp.csv"))
    print(sc[order(tissue, min_FDR)][, head(.SD, 8), by = tissue])
  }
} else message("sccomp not installed — skipped")
message("done")
