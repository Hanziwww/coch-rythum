#!/usr/bin/env Rscript
# Module 4 / round 9 (D75, O15, I099) — robustness of the Milo neighbourhood rhythms (G08).
# Rhythmic neighbourhoods peaked near sampling times with very large amplitudes, i.e. possibly neighbourhoods dominated
# by one or two samples. Three checks on the primary (scANVI latent) neighbourhoods of 53_milo_rhythm.R:
#   (0) the neighbourhoods and neighbourhood x sample counts are rebuilt with the same code and seed, and the stored
#       statistics (F, p, peak, spatial FDR) are asserted to be reproduced;
#   (1) sample dominance: share of a neighbourhood's cells from its largest sample, and that share relative to the
#       sample's share of all cells (> 50 % flags a neighbourhood as dominated by one sample);
#   (2) leave-one-time-point-out (LOTO; one sample per ZT): the edgeR QL cosinor refit without each sample, with the
#       Tier rules of 33_rhythm_tiers.py: single-sample-driven = p < 0.05 & (LOTO max p > 0.25 | LOTO min amplitude
#       ratio < 0.5); robust = spatial FDR < 0.1 & LOTO phase s.d. < 3 h & not single-sample-driven;
#   (3) exact time-relabelling null: the whole test (dispersion, QL F, spatial FDR) repeated for the 59 non-identity
#       dihedral relabelling classes of the six sampling times; per tissue and majority cell type the number of
#       neighbourhoods with spatial FDR < 0.1 (and of robust ones) is compared with the null (exact p = share of the 60
#       classes, observed included, with at least as many).
# Outputs (results/53_milo/): milo_counts_<tissue>.csv (neighbourhood x sample counts), milo_robustness_<tissue>.csv,
# milo_relabel_null.csv, milo_robustness_summary.csv
suppressPackageStartupMessages({ library(BiocNeighbors); library(edgeR); library(data.table); library(Matrix)
  library(matrixStats) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "53_milo")
K <- 30
TISSUES <- strsplit(Sys.getenv("TISSUES", "bm,cochlea"), ",")[[1]]
LAT <- file.path(RESULTS, "20_dr/embedding_%s_latent.csv")
classes <- dihedral_classes(6)
NCORES <- as.integer(Sys.getenv("NCORES", "8"))
stopifnot(length(classes) == 60, identical(as.integer(classes[[1]]), 0:5))

circ_sd_h <- function(peaks) {                       # circular s.d. of peak times (h)
  a <- 2 * pi * peaks / 24
  r <- sqrt(mean(cos(a))^2 + mean(sin(a))^2)
  sqrt(-2 * log(max(min(r, 1), 1e-12))) * 24 / (2 * pi)
}

fit_rhythm <- function(cnt, lib, zt) {                # the test of 53_milo_rhythm.R
  X <- cosinor_design(zt)
  y <- DGEList(cnt, lib.size = lib)
  y <- estimateDisp(y, X)
  qt <- glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)
  co <- qt$coefficients
  ap <- amp_phase(co[, 2], co[, 3])
  list(F = qt$table$F, p = qt$table$PValue, amp = ap$amp, peak = ap$peak)
}

spatial_fdr <- function(p, w) {
  o <- order(p)
  s <- numeric(length(p))
  s[o] <- pmin(1, rev(cummin(rev(sum(w) * p[o] / cumsum(w[o])))))
  s
}

null_rows <- list(); summ <- list()
for (tis in TISSUES) {
  # ---- (0) rebuild exactly as 53_milo_rhythm.R
  set.seed(1)
  emb <- fread(sprintf(LAT, tis))
  obs <- as.data.table(arrow::read_parquet(file.path(CACHE, sprintf("20_dr/%s/obs.parquet", tis))), keep.rownames = FALSE)
  stopifnot(nrow(obs) == nrow(emb))
  Z <- as.matrix(emb[, -1]); rownames(Z) <- emb$cell_id
  n <- nrow(Z)
  knn <- findKNN(Z, k = K, BNPARAM = KmknnParam())
  rv <- sample.int(n, round((if (tis == "cochlea") 0.05 else 0.1) * n))
  med <- t(vapply(rv, function(i) colMedians(Z[knn$index[i, ], , drop = FALSE]), numeric(ncol(Z))))
  idx <- unique(queryKNN(Z, med, k = 1, BNPARAM = KmknnParam())$index[, 1])
  nh <- sparseMatrix(i = c(idx, as.vector(t(knn$index[idx, ]))), j = c(seq_along(idx), rep(seq_along(idx), each = K)),
                     x = 1, dims = c(n, length(idx)))
  nh@x[] <- 1
  samples <- sort(unique(obs$sample_id))
  S <- sparseMatrix(i = seq_len(n), j = match(obs$sample_id, samples), x = 1, dims = c(n, length(samples)),
                    dimnames = list(NULL, samples))
  cnt <- as.matrix(crossprod(nh, S))
  smp <- unique(obs[, .(sample_id, zt)])[match(colnames(cnt), sample_id)]
  zt <- smp$zt
  keep <- rowSums(cnt) >= 10
  lib <- colSums(cnt)
  ck <- cnt[keep, , drop = FALSE]
  f0 <- fit_rhythm(ck, lib, zt)
  w <- 1 / knn$distance[idx[keep], K]
  sf0 <- spatial_fdr(f0$p, w)
  stored <- fread(file.path(out_dir, sprintf("milo_da_results_%s.csv", tis)))
  stopifnot(nrow(stored) == sum(keep), all(stored$nhood == which(keep)))
  stopifnot(max(abs(stored$F - f0$F)) < 1e-6 * max(1, max(abs(stored$F))), max(abs(stored$p - f0$p)) < 1e-9,
            max(abs(stored$spatial_fdr - sf0)) < 1e-9)
  message(sprintf("%s: reproduced %d tested neighbourhoods (spatial FDR < 0.1: %d)", tis, sum(keep), sum(sf0 < 0.1)))
  fwrite(data.table(tissue = tis, nhood = rep(which(keep), ncol(ck)), sample_id = rep(colnames(ck), each = nrow(ck)),
                    zt = rep(zt, each = nrow(ck)), n_cells = as.vector(ck)),
         file.path(out_dir, sprintf("milo_counts_%s.csv", tis)))
  # ---- (1) sample dominance
  share_all <- colSums(S) / n
  top <- max.col(ck, ties.method = "first")
  top_share <- rowMaxs(ck) / rowSums(ck)
  rel <- top_share / share_all[top]
  # ---- (2) LOTO refits
  lp <- la <- lk <- matrix(NA_real_, nrow(ck), ncol(ck))
  for (j in seq_len(ncol(ck))) {
    fj <- fit_rhythm(ck[, -j, drop = FALSE], lib[-j], zt[-j])
    lp[, j] <- fj$p; la[, j] <- fj$amp / pmax(f0$amp, 1e-12); lk[, j] <- fj$peak
  }
  loto_max_p <- rowMaxs(lp)
  loto_min_amp <- rowMins(la)
  loto_sd <- apply(lk, 1, circ_sd_h)
  ssd <- f0$p < 0.05 & (loto_max_p > 0.25 | loto_min_amp < 0.5)
  robust <- sf0 < 0.1 & loto_sd < 3 & !ssd
  rob <- data.table(tissue = tis, nhood = which(keep), majority_cell_type = stored$majority_cell_type, size = rowSums(ck),
                    peak_zt = f0$peak, amp_log2 = f0$amp / log(2), p = f0$p, spatial_fdr = sf0,
                    top_sample = colnames(ck)[top], top_sample_zt = zt[top], top_sample_share = top_share,
                    top_share_over_expected = rel, dominated = top_share > 0.5, n_samples_with_cells = rowSums(ck > 0),
                    loto_max_p = loto_max_p, loto_min_amp_ratio = loto_min_amp, loto_phase_sd_h = loto_sd,
                    single_sample_driven = ssd, robust = robust)
  fwrite(rob, file.path(out_dir, sprintf("milo_robustness_%s.csv", tis)))
  # ---- (3) exact relabelling null of the whole test (classes 2-60 in parallel; class 1 = observed)
  ord <- order(zt); zs <- zt[ord]
  one_class <- function(k) {
    ztk <- zt
    ztk[ord] <- zs[classes[[k]] + 1]
    fk <- fit_rhythm(ck, lib, ztk)
    callk <- spatial_fdr(fk$p, w) < 0.1
    rk <- rep(FALSE, length(callk))
    if (any(callk)) {                                 # LOTO rules under the relabelled times (all rows, as observed)
      lpk <- lak <- lkk <- matrix(NA_real_, nrow(ck), ncol(ck))
      for (j in seq_len(ncol(ck))) {
        fj <- fit_rhythm(ck[, -j, drop = FALSE], lib[-j], ztk[-j])
        lpk[, j] <- fj$p; lak[, j] <- fj$amp / pmax(fk$amp, 1e-12); lkk[, j] <- fj$peak
      }
      rk <- callk & apply(lkk, 1, circ_sd_h) < 3 & !(fk$p < 0.05 & (rowMaxs(lpk) > 0.25 | rowMins(lak) < 0.5))
    }
    data.table(majority_cell_type = stored$majority_cell_type, call = callk, robust = rk)
  }
  res_k <- c(list(data.table(majority_cell_type = stored$majority_cell_type, call = sf0 < 0.1, robust = robust)),
             parallel::mclapply(2:60, one_class, mc.cores = NCORES))
  stopifnot(length(res_k) == 60, !vapply(res_k, inherits, logical(1), "try-error"))
  for (k in seq_along(res_k)) {
    tab <- res_k[[k]][, .(n_sfdr10 = sum(call), n_robust = sum(robust)), by = majority_cell_type]
    tab <- rbind(tab, data.table(majority_cell_type = "ALL", n_sfdr10 = sum(res_k[[k]]$call),
                                 n_robust = sum(res_k[[k]]$robust)))
    null_rows[[length(null_rows) + 1]] <- tab[, `:=`(tissue = tis, relabel_class = k, observed = k == 1)]
  }
  message(sprintf("%s: robust %d of %d spatial-FDR calls; dominated among calls %.2f", tis, sum(robust), sum(sf0 < 0.1),
                  mean(top_share[sf0 < 0.1] > 0.5)))
}
nl <- rbindlist(null_rows, use.names = TRUE)
fwrite(nl, file.path(out_dir, "milo_relabel_null.csv"))
types <- unique(nl[, .(tissue, majority_cell_type)])
full <- CJ(tissue = unique(nl$tissue), majority_cell_type = unique(nl$majority_cell_type), relabel_class = 1:60)
full <- merge(full, nl[, .(tissue, majority_cell_type, relabel_class, n_sfdr10, n_robust)],
              by = c("tissue", "majority_cell_type", "relabel_class"), all.x = TRUE)
full[is.na(n_sfdr10), `:=`(n_sfdr10 = 0L, n_robust = 0L)]
full <- merge(full, types, by = c("tissue", "majority_cell_type"))
sm <- full[, {
  o1 <- n_sfdr10[relabel_class == 1]; o2 <- n_robust[relabel_class == 1]
  nn1 <- n_sfdr10[relabel_class > 1]; nn2 <- n_robust[relabel_class > 1]
  .(observed_sfdr10 = o1, null_mean_sfdr10 = mean(nn1), null_q95_sfdr10 = as.numeric(quantile(nn1, 0.95)),
    p_exact_sfdr10 = mean(n_sfdr10 >= o1), observed_robust = o2, null_mean_robust = mean(nn2),
    null_q95_robust = as.numeric(quantile(nn2, 0.95)), p_exact_robust = mean(n_robust >= o2))
}, by = .(tissue, majority_cell_type)]
fwrite(sm[order(tissue, -observed_sfdr10)], file.path(out_dir, "milo_robustness_summary.csv"))
print(sm[order(tissue, -observed_sfdr10)][observed_sfdr10 > 0 | majority_cell_type == "ALL"])
message("done")
