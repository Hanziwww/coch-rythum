#!/usr/bin/env Rscript
# Round 2, WP1 — multi-model robustness of every abundance rhythm (cell type × tissue).
# Models (24-h cosinor; ZT axis):
#   ols_pct          OLS on % of tissue cells (the user's figure model; from r2_figure/abundance_pct_rhythm.csv)
#   ols_logit        OLS on logit((n + 0.5) / (N + 1))
#   clr / alr_structural / alr_marrow / within_lineage   round-1 reference frames A–D (50_abundance), with their
#                    exact permutation p
#   propeller        logit proportions + limma robust eBayes (51)
#   sccomp           Bayesian beta-binomial with outlier detection (51; min FDR of the cos/sin effects)
#   quasibinomial    GLM on counts with quasi-binomial dispersion, F test of cos/sin (2 df)
#   betabinomial     GLM on counts (glmmTMB betabinomial), likelihood-ratio test of cos/sin (2 df)
#   marrow_adjusted  cochlear immune types: residual rhythm after the marrow-content decomposition (52; posterior
#                    probability that the residual peak-to-trough fold change exceeds 1.5)
# Core evidence set = {clr permutation p <= 0.05, propeller p < 0.05, sccomp FDR < 0.05, quasibinomial p < 0.05,
# betabinomial p < 0.05}. Verdict: robust (>= 4 core, LOTO phase SD < 3 h and circular SD of the core peaks < 3 h),
# supported (2–3 core), fragile (OLS-% p < 0.05 but <= 1 core), not rhythmic (otherwise).
# Outputs (results/r2_figure/): abundance_multimodel_long.csv, abundance_multimodel_summary.csv
suppressPackageStartupMessages({ library(data.table); library(glmmTMB) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "r2_figure")
ols <- fread(file.path(out_dir, "abundance_pct_rhythm.csv"))
ct <- fread(file.path(RESULTS, "00_audit/celltype_counts_by_sample.csv"))
smp <- unique(ct[, .(sample_id, tissue, zt)])
full <- merge(smp[, .(sample_id, tissue, zt)][, k := 1], unique(ct[, .(tissue, cell_type)])[, k := 1],
              by = c("tissue", "k"), allow.cartesian = TRUE)[, k := NULL]
full <- merge(full, ct[, .(sample_id, cell_type, n_cells)], by = c("sample_id", "cell_type"), all.x = TRUE)
full[is.na(n_cells), n_cells := 0]
full[, N := sum(n_cells), by = sample_id]
fr <- fread(file.path(RESULTS, "50_abundance/abundance_rhythm_frames.csv"))
pr <- fread(file.path(RESULTS, "51_abundance_models/abundance_rhythm_propeller.csv"))
sc <- fread(file.path(RESULTS, "51_abundance_models/abundance_rhythm_sccomp.csv"))
md <- fread(file.path(RESULTS, "52_marrow_decomposition/tissue_residual_rhythm.csv"))[model == "marrow_adjusted"]

long <- list()
addrow <- function(...) long[[length(long) + 1]] <<- data.table(...)
for (i in seq_len(nrow(ols))) {
  tis <- ols$tissue[i]; cty <- ols$cell_type[i]
  addrow(tissue = tis, cell_type = cty, model = "ols_pct", p = ols$p_ols[i], peak_zt = ols$peak_zt[i],
         stat = ols$F[i], note = "figure model")
  d <- full[tissue == tis & cell_type == cty][order(zt)]
  cv <- cos(W24 * d$zt); sv <- sin(W24 * d$zt)
  lg <- qlogis((d$n_cells + 0.5) / (d$N + 1))
  f1 <- lm(lg ~ cv + sv); a0 <- anova(lm(lg ~ 1), f1)
  ap <- amp_phase(coef(f1)[2], coef(f1)[3])
  addrow(tissue = tis, cell_type = cty, model = "ols_logit", p = a0$`Pr(>F)`[2], peak_zt = ap$peak, stat = a0$F[2], note = "")
  qb <- glm(cbind(n_cells, N - n_cells) ~ cv + sv, family = quasibinomial, data = d)
  qb0 <- glm(cbind(n_cells, N - n_cells) ~ 1, family = quasibinomial, data = d)
  aq <- anova(qb0, qb, test = "F")
  apq <- amp_phase(coef(qb)[2], coef(qb)[3])
  addrow(tissue = tis, cell_type = cty, model = "quasibinomial", p = aq$`Pr(>F)`[2], peak_zt = apq$peak,
         stat = aq$F[2], note = sprintf("dispersion %.1f", summary(qb)$dispersion))
  bb <- tryCatch(glmmTMB(cbind(n_cells, N - n_cells) ~ cv + sv, family = betabinomial, data = d), error = function(e) NULL)
  bb0 <- tryCatch(glmmTMB(cbind(n_cells, N - n_cells) ~ 1, family = betabinomial, data = d), error = function(e) NULL)
  if (!is.null(bb) && !is.null(bb0)) {
    lr <- 2 * (as.numeric(logLik(bb)) - as.numeric(logLik(bb0)))
    b <- fixef(bb)$cond; apb <- amp_phase(b["cv"], b["sv"])
    addrow(tissue = tis, cell_type = cty, model = "betabinomial", p = pchisq(lr, 2, lower.tail = FALSE),
           peak_zt = unname(apb$peak), stat = lr, note = "")
  }
  for (fname in unique(fr$frame)) {
    r <- fr[frame == fname & tissue == tis & feature == cty]
    if (!nrow(r)) next
    mname <- if (fname == "A_clr_all") "clr" else if (fname == "B_alr_structural") "alr_structural" else
      if (fname == "C_alr_marrow") "alr_marrow" else if (startsWith(fname, "D_within")) "within_lineage" else next
    addrow(tissue = tis, cell_type = cty, model = mname, p = r$p[1], peak_zt = r$peak_zt[1], stat = r$F[1],
           note = sprintf("%s; perm p %.3f; LOTO SD %.2f h", fname, r$p_perm[1], r$loto_phase_sd_h[1]))
  }
  r <- pr[tissue == tis & cell_type == cty]
  if (nrow(r)) addrow(tissue = tis, cell_type = cty, model = "propeller", p = r$p[1], peak_zt = r$peak_zt[1],
                      stat = r$F[1], note = sprintf("q %.3f", r$q_BH[1]))
  r <- sc[tissue == tis & cell_type == cty]
  if (nrow(r)) addrow(tissue = tis, cell_type = cty, model = "sccomp", p = r$min_FDR[1], peak_zt = r$peak_zt[1],
                      stat = r$amp_logit[1], note = "p column holds min FDR")
  r <- md[cell_type == cty]
  if (tis == "cochlea" && nrow(r)) addrow(tissue = tis, cell_type = cty, model = "marrow_adjusted",
                                          p = NA_real_, peak_zt = r$resid_peak_zt[1], stat = r$prob_fc_gt_1p5[1],
                                          note = "stat = P(residual FC > 1.5)")
}
long <- rbindlist(long, fill = TRUE)
long[, peak_clock := zt_to_clock(peak_zt)]
fwrite(long, file.path(out_dir, "abundance_multimodel_long.csv"))

clr_perm <- fr[frame == "A_clr_all", .(tissue, cell_type = feature, clr_p_perm = p_perm)]
summ <- long[, {
  core <- c(clr = NA, propeller = NA, sccomp = NA, quasibinomial = NA, betabinomial = NA)
  pk <- c()
  if (any(model == "propeller")) { core["propeller"] <- p[model == "propeller"][1] < 0.05; pk <- c(pk, peak_zt[model == "propeller"][1]) }
  if (any(model == "sccomp")) { core["sccomp"] <- p[model == "sccomp"][1] < 0.05; pk <- c(pk, peak_zt[model == "sccomp"][1]) }
  if (any(model == "quasibinomial")) { core["quasibinomial"] <- p[model == "quasibinomial"][1] < 0.05; pk <- c(pk, peak_zt[model == "quasibinomial"][1]) }
  if (any(model == "betabinomial")) { core["betabinomial"] <- p[model == "betabinomial"][1] < 0.05; pk <- c(pk, peak_zt[model == "betabinomial"][1]) }
  if (any(model == "clr")) pk <- c(pk, peak_zt[model == "clr"][1])
  list(n_models = .N, n_models_p05 = sum(p < 0.05, na.rm = TRUE), core_tested = sum(!is.na(core[-1])) ,
       core_supporting_ex_clr = sum(core[-1], na.rm = TRUE), core_peaks = paste(round(pk, 1), collapse = ";"),
       core_peak_circ_sd_h = if (length(pk) >= 2) circ_sd_h(matrix(pk, nrow = 1)) else NA_real_,
       ols_pct_p = p[model == "ols_pct"][1], ols_pct_peak_zt = peak_zt[model == "ols_pct"][1])
}, by = .(tissue, cell_type)]
summ <- merge(summ, clr_perm, by = c("tissue", "cell_type"), all.x = TRUE)
summ <- merge(summ, ols[, .(tissue, cell_type, mean_pct, q_BH_ols = q_BH, p_perm_ols = p_perm, loto_phase_sd_h,
                            boot_frac_p05, peak_ci_halfwidth_h)], by = c("tissue", "cell_type"))
summ[, core_supporting := core_supporting_ex_clr + as.integer(!is.na(clr_p_perm) & clr_p_perm <= 0.05)]
summ[, core_tested := core_tested + as.integer(!is.na(clr_p_perm))]   # CLR (frame A, permutation) is a core model
summ[, verdict := fifelse(core_supporting >= 4 & loto_phase_sd_h < 3 & (is.na(core_peak_circ_sd_h) | core_peak_circ_sd_h < 3),
                          "robust", fifelse(core_supporting >= 2, "supported",
                          fifelse(ols_pct_p < 0.05, "fragile", "not rhythmic")))]
summ[, ols_pct_peak_clock := zt_to_clock(ols_pct_peak_zt)]
fwrite(summ[order(tissue, factor(verdict, levels = c("robust", "supported", "fragile", "not rhythmic")), ols_pct_p)],
       file.path(out_dir, "abundance_multimodel_summary.csv"))
print(summ[verdict != "not rhythmic", .(tissue, cell_type, ols_pct_p = signif(ols_pct_p, 2), peak_zt = round(ols_pct_peak_zt, 1),
                                        core_supporting, core_peak_circ_sd_h = round(core_peak_circ_sd_h, 2), verdict)])
message("done")
