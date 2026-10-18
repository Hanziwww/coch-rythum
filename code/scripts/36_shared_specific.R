#!/usr/bin/env Rscript
# Module 2.7 — shared vs cell-type-specific rhythm components across cell types within each tissue.
# (1) dream (variancePartition) per gene on stacked pseudobulk of all included cell types of a tissue:
#       ~ group + cos + sin + group:cos + group:sin + (1 | sample_id), sum-to-zero contrasts for group
#     cos/sin = rhythm shared by all cell types (tested against between-sample variation, 3 df — a shared
#     component cannot be distinguished from a sinusoidal sample-level artefact); group:cos/sin = cell-type-specific
#     deviations (tested against within-sample variation across cell types — robust to sample-level artefacts).
#     Genes: expressed (filter of 31) in >= 50 % of the tissue's cell types.
# (2) mashr across cell types on the primary limma (a, b) estimates: posterior effects, lfsr, and pairwise
#     sharing of rhythms between cell types (both lfsr < 0.05 and posterior peaks within 3 h).
# Outputs (results/36_shared_specific/): dream_shared_specific_<tissue>.csv, mash_posterior_<tissue>.csv,
#   mash_pairwise_sharing_<tissue>.csv, mash_mixture_weights_<tissue>.csv, shared_specific_summary.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table); library(variancePartition)
  library(BiocParallel); library(mashr) })
source("scripts/lib/cosinor.R")
NCORES <- as.integer(Sys.getenv("NCORES", "8"))
out_dir <- file.path(RESULTS, "36_shared_specific"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status == "primary"]
summ <- list()
for (tis in strsplit(Sys.getenv("TISSUES", "cochlea,bm"), ",")[[1]]) {
  dream_file <- file.path(out_dir, sprintf("dream_shared_specific_%s.csv", tis))
  pb <- read_pseudobulk(tis)
  groups <- incl[tissue == tis, group]
  cnts <- list(); metas <- list(); keeps <- list()
  for (g in groups) {
    gc <- group_counts(pb, design_dt, tis, g)
    keeps[[g]] <- expr_filter(gc$counts)
    cnts[[g]] <- gc$counts
    metas[[g]] <- data.table(group = g, sample_id = gc$sample_id, zt = gc$zt)
  }
  if (file.exists(dream_file)) {            # cached from an earlier run (dream is deterministic)
    res <- fread(dream_file)
    message(tis, ": dream results reused from ", dream_file)
  } else {
  K <- do.call(cbind, keeps)
  genes <- rownames(K)[rowMeans(K) >= 0.5]
  y <- DGEList(do.call(cbind, lapply(cnts, function(m) m[genes, ])))
  meta <- rbindlist(metas)
  colnames(y) <- paste(meta$group, meta$sample_id, sep = "|")
  y <- calcNormFactors(y)
  meta[, `:=`(group = factor(group), cosv = cos(W24 * zt), sinv = sin(W24 * zt))]
  meta <- as.data.frame(meta); rownames(meta) <- colnames(y)
  contrasts(meta$group) <- contr.sum(nlevels(meta$group))
  form <- ~ group + cosv + sinv + group:cosv + group:sinv + (1 | sample_id)
  param <- MulticoreParam(NCORES)
  vobj <- voomWithDreamWeights(y, form, meta, BPPARAM = param)
  fit <- dream(vobj, form, meta, BPPARAM = param)
  fit <- eBayes(fit)
  cf <- colnames(coef(fit))
  shared_t <- topTable(fit, coef = c("cosv", "sinv"), number = Inf, sort.by = "none")
  spec_coefs <- grep("^group[0-9]+:(cosv|sinv)$", cf, value = TRUE)
  spec_t <- topTable(fit, coef = spec_coefs, number = Inf, sort.by = "none")
  ap <- amp_phase(coef(fit)[, "cosv"], coef(fit)[, "sinv"])
  res <- data.table(tissue = tis, gene = rownames(fit), n_celltypes = length(groups),
                    shared_amp_log2 = ap$amp, shared_ptt_fc = 2^(2 * ap$amp), shared_peak_zt = ap$peak,
                    shared_F = shared_t$F, shared_p = shared_t$P.Value, shared_q = shared_t$adj.P.Val,
                    specific_F = spec_t$F, specific_p = spec_t$P.Value, specific_q = spec_t$adj.P.Val)
  fwrite(res, dream_file)
  }
  message(sprintf("%s: %d genes; shared q<0.1: %d; specific q<0.1: %d", tis, nrow(res), sum(res$shared_q < 0.1),
                  sum(res$specific_q < 0.1)))

  # ---- mashr across cell types on primary limma estimates
  lim <- fread(file.path(RESULTS, sprintf("31_rhythm_genes/rhythm_limma_cell_type_%s.csv", tis)))[group %in% groups]
  allg <- unique(lim$gene)
  Bh <- matrix(0, length(allg), 2 * length(groups), dimnames = list(allg, c(paste0(groups, "|a"), paste0(groups, "|b"))))
  Sh <- matrix(1e3, length(allg), 2 * length(groups), dimnames = dimnames(Bh))
  for (g in groups) {
    l <- lim[group == g]
    i <- match(l$gene, allg)
    Bh[i, paste0(g, "|a")] <- l$a; Bh[i, paste0(g, "|b")] <- l$b
    Sh[i, paste0(g, "|a")] <- l$se_a; Sh[i, paste0(g, "|b")] <- l$se_b
  }
  # standard mashr workflow: strong set from condition-by-condition ash, mixture fitted on a random subset with
  # the null correlation across conditions estimated (cell types share samples), posteriors for all genes.
  # Covariances: identity, a rhythm shared by all cell types (common a and b), one cell-type-only rhythm per cell
  # type (arbitrary phase), and data-driven ED matrices from the strong set.
  set.seed(1)
  K <- length(groups); P <- 2 * K
  m1 <- mash_1by1(mash_set_data(Bh, Sh))
  strong <- get_significant_results(m1, thresh = 0.05)
  if (length(strong) < 200) strong <- order(apply(abs(Bh / Sh), 1, max), decreasing = TRUE)[1:500]
  random <- sample(seq_len(nrow(Bh)), min(4000, nrow(Bh)))
  Vhat <- estimate_null_correlation_simple(mash_set_data(Bh[random, ], Sh[random, ]))
  d_strong <- mash_set_data(Bh[strong, ], Sh[strong, ], V = Vhat)
  d_random <- mash_set_data(Bh[random, ], Sh[random, ], V = Vhat)
  U.c <- list(identity = diag(P))
  Ush <- matrix(0, P, P); Ush[1:K, 1:K] <- 1; Ush[K + 1:K, K + 1:K] <- 1
  U.c$shared_rhythm <- Ush
  for (k in seq_len(K)) { u <- matrix(0, P, P); u[k, k] <- 1; u[K + k, K + k] <- 1; U.c[[paste0("only_", groups[k])]] <- u }
  U.ed <- cov_ed(d_strong, cov_pca(d_strong, min(5, P - 1)))
  m_r <- mash(d_random, Ulist = c(U.c, U.ed), outputlevel = 1)
  m <- mash(mash_set_data(Bh, Sh, V = Vhat), g = get_fitted_g(m_r), fixg = TRUE)
  wts <- get_estimated_pi(m_r)
  fwrite(data.table(tissue = tis, component = names(wts), weight = as.numeric(wts)),
         file.path(out_dir, sprintf("mash_mixture_weights_%s.csv", tis)))
  message(tis, ": mash strong set ", length(strong), "; top components: ",
          paste(head(names(sort(wts, decreasing = TRUE)), 5), collapse = ", "))
  pm <- get_pm(m); lf <- get_lfsr(m)
  post <- rbindlist(lapply(groups, function(g) {
    a <- pm[, paste0(g, "|a")]; b <- pm[, paste0(g, "|b")]; ap2 <- amp_phase(a, b)
    data.table(tissue = tis, gene = allg, cell_type = g, post_a = a, post_b = b, post_ptt_fc = 2^(2 * ap2$amp),
               post_peak_zt = ap2$peak, lfsr_min = pmin(lf[, paste0(g, "|a")], lf[, paste0(g, "|b")]),
               tested = Sh[, paste0(g, "|a")] < 1e3)
  }))[tested == TRUE]
  fwrite(post, file.path(out_dir, sprintf("mash_posterior_%s.csv", tis)))
  sig <- post[lfsr_min < 0.05]
  pairs <- CJ(c1 = groups, c2 = groups)[c1 < c2]
  share <- pairs[, {
    s1 <- sig[cell_type == c1]; s2 <- sig[cell_type == c2]
    both <- merge(s1[, .(gene, p1 = post_peak_zt)], s2[, .(gene, p2 = post_peak_zt)], by = "gene")
    .(n_sig_1 = nrow(s1), n_sig_2 = nrow(s2), n_both = nrow(both),
      n_both_same_phase = sum(abs(circ_diff(both$p1, both$p2)) < 3),
      jaccard = nrow(both) / max(1, length(union(s1$gene, s2$gene))))
  }, by = .(c1, c2)]
  share[, tissue := tis]
  fwrite(share, file.path(out_dir, sprintf("mash_pairwise_sharing_%s.csv", tis)))
  summ[[tis]] <- data.table(tissue = tis, n_genes_dream = nrow(res), dream_shared_q10 = sum(res$shared_q < 0.1),
                            dream_specific_q10 = sum(res$specific_q < 0.1),
                            mash_sig_gene_celltype = nrow(sig), mash_genes_sig_any = uniqueN(sig$gene))
}
sf <- file.path(out_dir, "shared_specific_summary.csv")
old <- if (file.exists(sf)) fread(sf)[!tissue %in% names(summ)] else NULL
fwrite(rbindlist(c(list(old), summ), fill = TRUE), sf)
print(rbindlist(summ))
message("done")
