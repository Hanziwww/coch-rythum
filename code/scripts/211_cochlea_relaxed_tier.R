#!/usr/bin/env Rscript
# Round 2, WP2 — calibrated relaxed rhythm tier for the cochlea ("Tier C").
# Common base per gene × cochlear cell type: limma p < 0.05 & peak-to-trough FC >= 1.25 & LOTO phase SD < 3 h & not
# single-sample-driven (as in 33: LOTO max p <= 0.25 and min amplitude ratio >= 0.5). Supports:
#   s_edgeR    edgeR-QL (NB GLM) p < 0.05 on the same counts (a second model, not independent data);
#   s_external rhythmic in the external cochlea (GSE88954 sham, limma p < 0.05) with clock-aligned peaks within 4 h;
#   s_bm       rhythmic in the same cell type in temporal-bone marrow (p < 0.05) with clock-aligned peaks within 4 h.
# Clock alignment: internal phase = peak − offset of the tissue / dataset (r2_phase/clock_alignment_offsets.csv).
# Rule ladder (declared before calibration; R0 is the pre-specified rule of the approved plan):
#   R0 base & (s_edgeR | s_external | s_bm)          R1 base & (s_external | s_bm)   [independent-data support]
#   R2 R0 & p < 0.01                                  R3 R1 & p < 0.01
#   R4 base & p < 0.01 & FC >= 1.5 & LOTO SD < 2 & s_edgeR   [strict, internal evidence only]
# Calibration: every rule — limma, edgeR-QL, LOTO and both phase supports recomputed — under each of the 59
# non-identity dihedral relabelings of the cochlear time labels (BM and external phases fixed) → expected false calls
# per cell type; empirical FDR = mean null calls / observed calls. Tier C (calibrated) = the rule with the most calls
# among those with overall empirical FDR <= 0.2 (target fixed in advance); if none qualifies, the lowest-FDR rule.
# Simulation (power of the internal criteria): rhythms injected (FC 1.5 / 2 / 3, 5 % of genes, binomial thinning) into
# 3 null-relabelled cell types; R0/R2/R4 power and FDP (external/BM supports cannot be simulated: they require real
# external rhythms). RAIN/JTK/LR p-values and the tissue-pattern score (210) are attached as annotations.
# Outputs (results/r2_cochlea/): cochlea_relaxed_tier.csv, cochlea_relaxed_calibration.csv,
#   cochlea_relaxed_rule_selection.csv, cochlea_relaxed_simulation.csv
# Round 3 (WP5): env TISSUE=bm applies the same ladder to temporal-bone marrow → results/r3_bm/bm_relaxed_*.csv.
#   BM supports: s_external = GSE198770 BM control fitted on replicates 1–2 only (replicate 3 is held out for the
#   independence test in 341), s_cochlea = same cell type rhythmic in our cochlea; plus the BM tissue-level pseudobulk
#   cosinor (bm_tissue_pseudobulk_cosinor.csv). With the default TISSUE=cochlea the behaviour is unchanged.
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table); library(parallel) })
source("scripts/lib/cosinor.R")
NC <- as.integer(Sys.getenv("NCORES", "4"))
FDR_TARGET <- 0.2
RULES <- c("R0", "R1", "R2", "R3", "R4")
TISSUE <- Sys.getenv("TISSUE", "cochlea")
OTHER <- if (TISSUE == "cochlea") "bm" else "cochlea"
OTHER_LABEL <- if (OTHER == "bm") "BM" else "cochlea"
out_dir <- file.path(RESULTS, if (TISSUE == "cochlea") "r2_cochlea" else "r3_bm")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & tissue == TISSUE &
                                                                        status != "excluded"]
PB <- read_pseudobulk(TISSUE)
classes <- dihedral_classes(6)
off <- fread(file.path(RESULTS, "r2_phase/clock_alignment_offsets.csv"))
d_self <- off[dataset == "this_atlas" & tissue == TISSUE & unit == "tissue_pool", offset_canonical_h]
d_other <- off[dataset == "this_atlas" & tissue == OTHER & unit == "tissue_pool", offset_canonical_h]
if (TISSUE == "cochlea") {
  d_ext <- off[dataset == "GSE88954" & condition == "sham", offset_canonical_h][1]
  ext <- fread(file.path(RESULTS, "71_external/external_rhythm_GSE88954.csv"))[condition == "sham",
            .(gene, ext_p = p, ext_internal = (peak_h - d_ext) %% 24)]
} else {
  d_ext <- off[dataset == "GSE198770_bone_marrow" & condition == "control" & unit == "bulk", offset_canonical_h][1]
  em <- fread(file.path(ROOT, "external/prepared_r2/GSE198770_bone_marrow__meta.csv"))[condition == "control"]
  print(table(em$time_h, em$replicate))
  em <- em[replicate %in% c(1, 2)]                    # replicate 3 is held out for the independence test (341)
  ex <- fread(file.path(ROOT, "external/prepared_r2/GSE198770_bone_marrow__expr.csv"))
  E <- as.matrix(ex[, em$sample, with = FALSE]); rownames(E) <- ex[[1]]
  E <- log2(E + 1); E <- E[rowMeans(E) > 1 & !duplicated(rownames(E)), ]
  fx <- eBayes(lmFit(E, cosinor_design(em$time_h)), trend = TRUE, robust = TRUE)
  tx <- topTable(fx, coef = 2:3, number = Inf, sort.by = "none")
  ax <- amp_phase(fx$coefficients[, 2], fx$coefficients[, 3])
  ext <- data.table(gene = rownames(E), ext_p = tx$P.Value, ext_internal = (ax$peak - d_ext) %% 24)
}
message(sprintf("offsets (h): %s %.2f, %s %.2f, external %.2f", TISSUE, d_self, OTHER, d_other, d_ext))
ext <- ext[!duplicated(gene)]
other <- fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", OTHER)),
               select = c("group", "gene", "p", "peak_zt"))[, .(group, gene, oth_p = p, oth_internal = (peak_zt - d_other) %% 24)]
ann <- fread(file.path(RESULTS, sprintf("33_rhythm_tiers/rhythm_genes_all_cell_type_%s.csv", TISSUE)),
             select = c("group", "gene", "rain_p", "jtk_p", "tier", "q_BH", "AveExpr"))
lr <- fread(file.path(RESULTS, sprintf("32_rhythm_alt/rhythm_lr_cell_type_%s.csv", TISSUE)), select = c("group", "gene", "lr_p"))
pat <- fread(file.path(RESULTS, sprintf("r2_cochlea/gene_pattern_scores_%s.csv", TISSUE)))[, .(group = cell_type, gene, pattern_r)]

eval_rule <- function(cnt, zt, nf, g) {
  r <- limma_cosinor(cnt, zt, norm_factors = nf)$table
  ap <- amp_phase(r$a, r$b)
  r[, `:=`(amp = ap$amp, peak_zt = ap$peak, ptt_fc = 2^(2 * ap$amp))]
  lo_pk <- lo_amp <- lo_p <- matrix(NA_real_, nrow(r), length(zt))
  for (j in seq_along(zt)) {
    rr <- limma_cosinor(cnt[, -j, drop = FALSE], zt[-j], norm_factors = nf[-j])$table
    a2 <- amp_phase(rr$a, rr$b)
    lo_pk[, j] <- a2$peak; lo_amp[, j] <- a2$amp; lo_p[, j] <- rr$p
  }
  r[, `:=`(loto_sd = circ_sd_h(lo_pk), loto_max_p = apply(lo_p, 1, max),
           loto_min_ratio = apply(lo_amp, 1, min) / pmax(amp, 1e-12))]
  y <- DGEList(cnt); y$samples$norm.factors <- nf
  X <- cosinor_design(zt)
  y <- estimateDisp(y, X)
  r[, edgeR_p := glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)$table$PValue]
  r[, group := g]
  r <- merge(r, ext, by = "gene", all.x = TRUE)
  r <- merge(r, other[group == g, .(gene, oth_p, oth_internal)], by = "gene", all.x = TRUE)
  r[, internal := (peak_zt - d_self) %% 24]
  r[, `:=`(s_edgeR = edgeR_p < 0.05,
           s_external = !is.na(ext_p) & ext_p < 0.05 & abs(circ_diff(internal, ext_internal)) < 4,
           s_other = !is.na(oth_p) & oth_p < 0.05 & abs(circ_diff(internal, oth_internal)) < 4)]
  r[, single_driven := p < 0.05 & (loto_max_p > 0.25 | loto_min_ratio < 0.5)]
  r[, base := p < 0.05 & ptt_fc >= 1.25 & loto_sd < 3 & !single_driven]
  r[, `:=`(R0 = base & (s_edgeR | s_external | s_other), R1 = base & (s_external | s_other),
           R2 = base & p < 0.01 & (s_edgeR | s_external | s_other), R3 = base & p < 0.01 & (s_external | s_other),
           R4 = base & p < 0.01 & ptt_fc >= 1.5 & loto_sd < 2 & s_edgeR)]
  for (k in c("base", RULES)) set(r, which(is.na(r[[k]])), k, FALSE)
  r
}

run_group <- function(g) {
  gc <- group_counts(PB, design_dt, TISSUE, g)
  cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
  nf <- calcNormFactors(DGEList(cnt))$samples$norm.factors
  obs <- eval_rule(cnt, gc$zt, nf, g)
  pg <- pat[group == g]
  obs <- merge(obs, pg[, .(gene, pattern_r)], by = "gene", all.x = TRUE)
  pattern_like <- setNames(!is.na(obs$pattern_r) & obs$pattern_r > 0.8, obs$gene)
  null <- rbindlist(lapply(2:length(classes), function(k) {
    nr <- eval_rule(cnt, gc$zt[classes[[k]] + 1], nf, g)
    pl <- pattern_like[nr$gene]
    data.table(class = k, rule = RULES, calls = vapply(RULES, function(x) sum(nr[[x]]), 0),
               calls_no_pattern = vapply(RULES, function(x) sum(nr[[x]] & !pl), 0))
  }))
  cal <- rbindlist(lapply(RULES, function(x) {
    nl <- null[rule == x]
    o <- sum(obs[[x]]); onp <- sum(obs[[x]] & !pattern_like[obs$gene])
    data.table(group = g, rule = x, n_tested = nrow(obs), observed_calls = o, observed_calls_no_pattern = onp,
               null_mean = mean(nl$calls), null_q95 = unname(quantile(nl$calls, 0.95)),
               null_mean_no_pattern = mean(nl$calls_no_pattern),
               perm_p_count = (1 + sum(nl$calls >= o)) / length(classes))
  }))
  message(g, " done (R0 observed ", sum(obs$R0), ")")
  list(obs = obs, cal = cal)
}
res <- mclapply(incl$group, function(g) tryCatch(run_group(g), error = function(e) { message(g, ": ", conditionMessage(e)); NULL }),
                mc.cores = NC)
res <- res[!vapply(res, is.null, TRUE)]
cal <- rbindlist(lapply(res, `[[`, "cal"))
tot <- cal[, .(group = "ALL", n_tested = sum(n_tested), observed_calls = sum(observed_calls),
               observed_calls_no_pattern = sum(observed_calls_no_pattern), null_mean = sum(null_mean),
               null_q95 = NA_real_, null_mean_no_pattern = sum(null_mean_no_pattern), perm_p_count = NA_real_), by = rule]
cal <- rbind(cal, tot, use.names = TRUE)
cal[, `:=`(empirical_fdr = pmin(1, null_mean / pmax(observed_calls, 1)),
           empirical_fdr_no_pattern = pmin(1, null_mean_no_pattern / pmax(observed_calls_no_pattern, 1)))]
all_ <- cal[group == "ALL"][order(match(rule, RULES))]
ok <- all_[empirical_fdr <= FDR_TARGET]
sel <- if (nrow(ok)) ok[which.max(observed_calls), rule] else all_[which.min(empirical_fdr), rule]
selection <- all_[, .(rule, definition = c(R0 = sprintf("base & (edgeR | external | %s support) [pre-specified]", OTHER_LABEL),
                                           R1 = sprintf("base & (external | %s support)", OTHER_LABEL), R2 = "R0 & p < 0.01", R3 = "R1 & p < 0.01",
                                           R4 = "base & p < 0.01 & FC >= 1.5 & LOTO SD < 2 & edgeR")[rule],
                      observed_calls, null_mean, empirical_fdr, observed_calls_no_pattern, empirical_fdr_no_pattern,
                      fdr_target = FDR_TARGET, selected = rule == sel)]
message("selected rule: ", sel)

tab <- rbindlist(lapply(res, `[[`, "obs"), fill = TRUE)
tab <- merge(tab, ann, by = c("group", "gene"), all.x = TRUE, suffixes = c("", "_primary"))
tab <- merge(tab, lr, by = c("group", "gene"), all.x = TRUE)
tab[, `:=`(tissue_pattern_like = !is.na(pattern_r) & pattern_r > 0.8, peak_clock = zt_to_clock(peak_zt),
           tierC_prespecified = R0, tierC = get(sel), rule_selected = sel)]
ct_fdr <- cal[rule == sel & group != "ALL", .(group, celltype_empirical_fdr = empirical_fdr)]
tab <- merge(tab, ct_fdr, by = "group", all.x = TRUE)
setnames(tab, "group", "cell_type")
setnames(tab, c("s_other", "oth_p", "oth_internal"), paste0(c("s_", "", ""), OTHER, c("", "_p", "_internal")))
setnames(cal, "group", "cell_type")
fwrite(tab[order(cell_type, -tierC, p)], file.path(out_dir, sprintf("%s_relaxed_tier.csv", TISSUE)))
fwrite(cal, file.path(out_dir, sprintf("%s_relaxed_calibration.csv", TISSUE)))
fwrite(selection, file.path(out_dir, sprintf("%s_relaxed_rule_selection.csv", TISSUE)))
print(selection)
print(cal[rule == sel][order(-observed_calls)][1:12])

# ---- simulation: power / FDP of the internal criteria (R0, R2, R4) on null-relabelled data with injected rhythms
libs <- sapply(incl[status == "primary", group], function(g) sum(group_counts(PB, design_dt, TISSUE, g)$counts))
pick <- names(sort(libs))[round(quantile(seq_along(libs), c(0.25, 0.6, 0.95)))]
jobs <- list()
for (g in pick) for (fc in c(1.5, 2, 3)) jobs[[length(jobs) + 1]] <- list(g = g, fc = fc, seed = 100 * match(g, pick) + 10 * fc)
sim <- rbindlist(mclapply(jobs, function(j) {
  set.seed(j$seed)
  gc <- group_counts(PB, design_dt, TISSUE, j$g)
  cnt <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
  k <- sample(2:length(classes), 1)
  zt <- gc$zt[classes[[k]] + 1]
  inj <- sample(nrow(cnt), round(0.05 * nrow(cnt)))
  phi <- runif(length(inj), 0, 24)
  f <- 2^(log2(j$fc) / 2 * cos(W24 * outer(phi, zt, function(p, t) t - p)))
  f <- f / apply(f, 1, max)
  cnt[inj, ] <- matrix(rbinom(length(f), as.vector(cnt[inj, ]), as.vector(f)), nrow = length(inj))
  storage.mode(cnt) <- "integer"
  nf <- calcNormFactors(DGEList(cnt))$samples$norm.factors
  r <- eval_rule(cnt, zt, nf, j$g)
  is_inj <- r$gene %in% rownames(cnt)[inj]
  rbindlist(lapply(c("base", "R0", "R2", "R4"), function(x) data.table(
    cell_type = j$g, relabel_class = paste(classes[[k]], collapse = ""), ptt_fc_injected = j$fc, rule = x,
    n_injected = sum(is_inj), n_calls = sum(r[[x]]), power = mean(r[[x]][is_inj]),
    fdp = if (sum(r[[x]])) sum(r[[x]] & !is_inj) / sum(r[[x]]) else NA_real_)))
}, mc.cores = NC))
fwrite(sim, file.path(out_dir, sprintf("%s_relaxed_simulation.csv", TISSUE)))
print(sim[, .(power = round(mean(power), 3), fdp = round(mean(fdp, na.rm = TRUE), 3)), by = .(rule, ptt_fc_injected)])

# ---- tissue-level pseudobulk cosinor (all cell types summed per sample) for selection-free external predictions
if (TISSUE != "cochlea") {                            # the cochlear version is r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv
  tot <- NULL
  for (g in incl$group) {
    gc <- group_counts(PB, design_dt, TISSUE, g)
    tot <- if (is.null(tot)) gc$counts else tot + gc$counts[rownames(tot), ]
    zt <- gc$zt
  }
  tot <- tot[expr_filter(tot), ]
  tp <- limma_cosinor(tot, zt)$table
  ap <- amp_phase(tp$a, tp$b)
  tp[, `:=`(peak_zt = ap$peak, amp_log2 = ap$amp, ptt_fc = 2^(2 * ap$amp))]
  fwrite(tp, file.path(out_dir, sprintf("%s_tissue_pseudobulk_cosinor.csv", TISSUE)))
  message(sprintf("tissue pseudobulk: %d genes, %d with p < 0.05", nrow(tp), sum(tp$p < 0.05)))
}
message("done")
