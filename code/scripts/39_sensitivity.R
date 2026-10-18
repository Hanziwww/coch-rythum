#!/usr/bin/env Rscript
# Module 2.3/2.4 — sensitivity analyses of the primary pseudobulk rhythm calls (31/33).
# (a) Depth matching: each cell type's 6 pseudobulk samples are binomially thinned to the smallest library among them
#     (3 replicate thinnings, seeds 1–3), then refitted with the primary limma-voom cosinor. Sequencing depth varies
#     ~2× between samples and is itself time-structured in the cochlea, so genes whose rhythm survives depth matching
#     are not explained by depth-dependent transformation/composition effects. Also: phase concentration (circular
#     mean, resultant length) of q < 0.1 genes before vs after matching.
# (b) Within-type state mix (immune types): per-sample mean of the cell type's state score (66_state_continua:
#     neutrotime for granulocytic types, b_maturation for B-lineage, mono_to_mac for monocytes/macrophages) added as a
#     covariate (~ cos + sin + state; 2 residual df, robust eBayes) — guards against rhythms created by a shifting
#     sub-state composition within a gold-standard label. Sensitivity only.
# (c) Cell-subsampling pseudo-replicates (Xu 2024; anticonservative by construction because the 3 pseudo-replicates of
#     a time point come from the same pooled sample): Tier-1 + clock genes, cells exported by 38_export_cells.py
#     (<= 300 per cell type × sample), each sample split at random into 3 parts → 18 pseudobulks, voom with the cells'
#     full library sizes (limma-trend on logCPM when < 50 genes).
# Outputs (results/39_sensitivity/): depth_matched_summary.csv, depth_matched_genes_<tissue>.csv,
#   statemix_sensitivity.csv, statemix_summary.csv, pseudoreplicate_rhythm.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "39_sensitivity"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status != "excluded"]

circ_mean <- function(h) { z <- complex(argument = W24 * h); m <- mean(z); c((Arg(m) / W24) %% 24, Mod(m)) }
pi1 <- function(p) 1 - limma::propTrueNull(p, method = "lfdr")

dm_summ <- list(); sm_rows <- list(); sm_summ <- list(); pr_rows <- list()
for (tis in c("cochlea", "bm")) {
  pb <- read_pseudobulk(tis)
  tiers <- fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", tis)),
                 select = c("group", "gene", "tier", "p", "q_BH", "peak_zt", "amp_log2"))
  dm_genes <- list()
  for (g in incl[tissue == tis, group]) {
    gc <- group_counts(pb, design_dt, tis, g)
    cnt <- gc$counts
    prim <- tiers[group == g]
    pk <- prim[q_BH < 0.1, peak_zt]
    cm0 <- if (length(pk)) circ_mean(pk) else c(NA, NA)
    dm_summ[[length(dm_summ) + 1]] <- data.table(tissue = tis, cell_type = g, version = "primary", replicate = 0,
      lib_min = min(colSums(cnt)), lib_max = max(colSums(cnt)), n_tested = nrow(prim),
      n_p01 = sum(prim$p < 0.01), n_q05 = sum(prim$q_BH < 0.05), n_q10 = sum(prim$q_BH < 0.1), pi1 = pi1(prim$p),
      q10_phase_mean_zt = cm0[1], q10_phase_resultant = cm0[2], frac_primary_q10_p05 = NA_real_,
      frac_tier12_p05 = NA_real_)
    # ---- (a) depth matching
    lib0 <- colSums(cnt)
    prob <- min(lib0) / lib0
    for (r in 1:3) {
      set.seed(r)
      thin <- matrix(rbinom(length(cnt), as.vector(cnt), rep(prob, each = nrow(cnt))), nrow(cnt),
                     dimnames = dimnames(cnt))
      keep <- expr_filter(thin)
      if (sum(keep) < 100) next
      res <- limma_cosinor(thin[keep, , drop = FALSE], gc$zt)$table
      ap <- amp_phase(res$a, res$b)
      res[, `:=`(amp_log2 = ap$amp, peak_zt = ap$peak)]
      pk <- res[q_BH < 0.1, peak_zt]
      cm <- if (length(pk)) circ_mean(pk) else c(NA, NA)
      m <- merge(prim[, .(gene, tier, q_prim = q_BH)], res[, .(gene, p_dm = p)], by = "gene")
      dm_summ[[length(dm_summ) + 1]] <- data.table(tissue = tis, cell_type = g, version = "depth_matched",
        replicate = r, lib_min = min(lib0), lib_max = max(lib0), n_tested = nrow(res), n_p01 = sum(res$p < 0.01),
        n_q05 = sum(res$q_BH < 0.05), n_q10 = sum(res$q_BH < 0.1), pi1 = pi1(res$p),
        q10_phase_mean_zt = cm[1], q10_phase_resultant = cm[2],
        frac_primary_q10_p05 = if (any(m$q_prim < 0.1)) mean(m[q_prim < 0.1, p_dm] < 0.05) else NA_real_,
        frac_tier12_p05 = if (any(m$tier %in% c("tier1", "tier2"))) mean(m[tier %in% c("tier1", "tier2"), p_dm] < 0.05)
                          else NA_real_)
      if (r == 1) dm_genes[[g]] <- res[, .(tissue = tis, cell_type = g, gene, amp_log2_dm = amp_log2,
                                           peak_zt_dm = peak_zt, p_dm = p, q_dm = q_BH)]
    }
  }
  dg <- merge(rbindlist(dm_genes), tiers[, .(cell_type = group, gene, tier, p_primary = p, q_primary = q_BH,
                                             peak_zt_primary = peak_zt, amp_log2_primary = amp_log2)],
              by = c("cell_type", "gene"), all.x = TRUE)
  fwrite(dg, file.path(out_dir, sprintf("depth_matched_genes_%s.csv", tis)))
  message(tis, ": depth matching done")

  # ---- (b) within-type state-mix covariate
  ss <- fread(file.path(RESULTS, "66_state_continua/state_scores_by_sample.csv"))[tissue_code == tis]
  score_for <- function(ct) {
    if (grepl("^Neutrophil|Granulocyte-Monocyte", ct)) "neutrotime"
    else if (grepl("B Cell|Pre-B|Pro-B", ct)) "b_maturation"
    else if (grepl("^Monocyte|^Macrophage", ct)) "mono_to_mac" else NA_character_
  }
  for (g in incl[tissue == tis, group]) {
    sc <- score_for(g)
    if (is.na(sc)) next
    gc <- group_counts(pb, design_dt, tis, g)
    cv <- ss[cell_type == g & score == sc][match(gc$sample_id, sample_id), mean]
    if (any(is.na(cv)) || sd(cv) == 0) next
    cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
    y <- calcNormFactors(DGEList(cnt))
    X <- cbind(cosinor_design(gc$zt), state = as.numeric(scale(cv)))
    v <- voom(y, X)
    fit <- eBayes(lmFit(v, X), robust = TRUE)
    tt <- topTable(fit, coef = 2:3, number = Inf, sort.by = "none")
    ap <- amp_phase(fit$coefficients[, 2], fit$coefficients[, 3])
    st <- topTable(fit, coef = 4, number = Inf, sort.by = "none")
    r <- data.table(tissue = tis, cell_type = g, state_score = sc, gene = rownames(fit), amp_log2_adj = ap$amp,
                    peak_zt_adj = ap$peak, p_statemix_adj = tt$P.Value, q_statemix_adj = tt$adj.P.Val,
                    state_coef = fit$coefficients[, 4], p_state = st$P.Value,
                    state_time_r2 = summary(lm(cv ~ cos(W24 * gc$zt) + sin(W24 * gc$zt)))$r.squared)
    r <- merge(r, tiers[group == g, .(gene, tier, p_primary = p, peak_zt_primary = peak_zt,
                                      amp_log2_primary = amp_log2)], by = "gene", all.x = TRUE)
    sm_rows[[g]] <- r
    t12 <- r[tier %in% c("tier1", "tier2")]
    sm_summ[[length(sm_summ) + 1]] <- data.table(tissue = tis, cell_type = g, state_score = sc,
      state_time_r2 = r$state_time_r2[1], n_tier12 = nrow(t12),
      frac_tier12_p05_after = if (nrow(t12)) mean(t12$p_statemix_adj < 0.05) else NA_real_,
      median_amp_ratio = if (nrow(t12)) median(t12$amp_log2_adj / t12$amp_log2_primary) else NA_real_,
      median_abs_peak_shift_h = if (nrow(t12)) median(abs(circ_diff(t12$peak_zt_adj, t12$peak_zt_primary))) else NA_real_)
  }
  message(tis, ": state-mix sensitivity done")

  # ---- (c) pseudo-replicates from exported cells (Tier-1 + clock genes)
  f <- file.path(CACHE, "38_cell_level", sprintf("cells_%s.csv.gz", tis))
  if (file.exists(f)) {
    d <- fread(f)
    genes <- setdiff(names(d), c("cell_type", "sample_id", "zt", "lib"))
    set.seed(1)
    for (g in unique(d$cell_type)) {
      sub <- d[cell_type == g]
      nps <- sub[, .N, by = sample_id]
      if (nrow(nps) < 6 || min(nps$N) < 30) next
      sub[, part := sample(rep_len(1:3, .N)), by = sample_id]
      agg <- sub[, c(lapply(.SD, sum), list(lib = sum(lib), zt = zt[1])), by = .(sample_id, part), .SDcols = genes]
      cnt <- t(as.matrix(agg[, ..genes]))
      X <- cosinor_design(agg$zt)
      if (length(genes) >= 50) {
        v <- voom(cnt, X, lib.size = agg$lib)
        fit <- eBayes(lmFit(v, X), robust = TRUE)
      } else {
        lc <- log2(t((t(cnt) + 0.5) / (agg$lib + 1)) * 1e6)
        fit <- eBayes(lmFit(lc, X), trend = TRUE, robust = TRUE)
      }
      tt <- topTable(fit, coef = 2:3, number = Inf, sort.by = "none")
      ap <- amp_phase(fit$coefficients[, 2], fit$coefficients[, 3])
      pr_rows[[paste(tis, g)]] <- merge(data.table(tissue = tis, cell_type = g, gene = rownames(fit),
                                                    n_cells = nrow(sub), amp_log2_pseudorep = ap$amp,
                                                    peak_zt_pseudorep = ap$peak, p_pseudorep = tt$P.Value,
                                                    q_pseudorep = tt$adj.P.Val),
                                         tiers[group == g, .(gene, tier, p_primary = p)], by = "gene", all.x = TRUE)
    }
    message(tis, ": pseudo-replicates done")
  }
}
fwrite(rbindlist(dm_summ), file.path(out_dir, "depth_matched_summary.csv"))
fwrite(rbindlist(sm_rows, fill = TRUE), file.path(out_dir, "statemix_sensitivity.csv"))
fwrite(rbindlist(sm_summ), file.path(out_dir, "statemix_summary.csv"))
pr <- rbindlist(pr_rows, fill = TRUE)
pr[, note := "anticonservative: pseudo-replicates share their pooled sample"]
fwrite(pr, file.path(out_dir, "pseudoreplicate_rhythm.csv"))
print(rbindlist(dm_summ)[, .(n_q10 = mean(n_q10), pi1 = mean(pi1), R = mean(q10_phase_resultant, na.rm = TRUE),
                              tier12_kept = mean(frac_tier12_p05, na.rm = TRUE)), by = .(tissue, version)])
print(rbindlist(sm_summ))
message("done")
