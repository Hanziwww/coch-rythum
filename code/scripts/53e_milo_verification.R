#!/usr/bin/env Rscript
# Round 9 — verification of the Milo exact-null result (I100; user: "verify whether that is really the case").
# Neighbourhoods are rebuilt exactly as in 53_milo_rhythm.R (stored p asserted) for both embeddings (primary scANVI;
# scVI sensitivity) and both tissues.
#   V1 audit: the cosinor F of every neighbourhood must be identical under the 12 rotations / reflections of the six
#      sampling times (the symmetry that defines the 60 relabelling classes), and the 60 class designs must differ.
#   V3 per-neighbourhood exact null: the rhythm test is refitted for all 60 classes (class 1 = observed); written per
#      neighbourhood x class (F, p, peak) for 53f_milo_verification.py, which ranks the observed class within each
#      neighbourhood and tests the best-class count / rank sum against the same 60 classes.
#   V2 positive control (primary embedding): smooth 24-h rhythms (peak-to-trough 2, 3, 5, 8-fold; 1 = no injection) are
#      injected into 5 % of the real neighbourhoods by binomial thinning (as 33b_simulation.R), aligned with the times of
#      a non-observed class k0; the whole test is refitted for all 60 classes and three statistics are ranked for k0:
#      number of spatial-FDR < 0.1 calls (the G08 statistic), number of neighbourhoods whose best class is k0, and the
#      rank sum of -log10 p.
#   Membership of the rhythmic neighbourhoods (spatial FDR < 0.1) is exported for the external check (V5).
# Outputs (results/53_milo/): milo_audit_symmetry.csv, milo_relabel_fits_<tissue>_<emb>.csv,
# milo_positive_control.csv, milo_membership_<tissue>.csv
suppressPackageStartupMessages({ library(BiocNeighbors); library(edgeR); library(data.table); library(Matrix)
  library(matrixStats) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "53_milo")
K <- 30
NCORES <- as.integer(Sys.getenv("NCORES", "12"))
classes <- dihedral_classes(6)
stopifnot(length(classes) == 60, identical(as.integer(classes[[1]]), 0:5))
EMB <- list(scanvi = list(lat = file.path(RESULTS, "20_dr/embedding_%s_latent.csv"), sfx = ""),
            scvi = list(lat = file.path(CACHE, "r2_embedding/latent_%s_scvi.csv"), sfx = "_scvi"))

fit_rhythm <- function(cnt, lib, zt) {
  X <- cosinor_design(zt)
  y <- DGEList(cnt, lib.size = lib)
  y <- estimateDisp(y, X)
  qt <- glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)
  co <- qt$coefficients
  list(F = qt$table$F, p = qt$table$PValue, peak = amp_phase(co[, 2], co[, 3])$peak)
}
spatial_fdr <- function(p, w) {
  o <- order(p); s <- numeric(length(p))
  s[o] <- pmin(1, rev(cummin(rev(sum(w) * p[o] / cumsum(w[o])))))
  s
}
class_times <- function(zt, k) {                      # times of class k (positions in ZT order permuted)
  ord <- order(zt); zs <- zt[ord]; z <- zt
  z[ord] <- zs[classes[[k]] + 1]
  z
}
build <- function(tis, emb) {
  set.seed(1)
  e <- fread(sprintf(EMB[[emb]]$lat, tis))
  obs <- as.data.table(arrow::read_parquet(file.path(CACHE, sprintf("20_dr/%s/obs.parquet", tis))), keep.rownames = FALSE)
  stopifnot(nrow(obs) == nrow(e))
  Z <- as.matrix(e[, -1]); rownames(Z) <- e$cell_id
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
  zt <- unique(obs[, .(sample_id, zt)])[match(colnames(cnt), sample_id)]$zt
  keep <- rowSums(cnt) >= 10
  list(Z = Z, knn = knn, idx = idx, nh = nh, cnt = cnt, ck = cnt[keep, , drop = FALSE], keep = keep, lib = colSums(cnt),
       zt = zt, w = 1 / knn$distance[idx[keep], K], cells = rownames(Z), obs = obs)
}

audit <- list(); pc_rows <- list()
for (tis in c("bm", "cochlea")) {
  for (emb in names(EMB)) {
    b <- build(tis, emb)
    f0 <- fit_rhythm(b$ck, b$lib, b$zt)
    stored <- fread(file.path(out_dir, sprintf("milo_da_results_%s%s.csv", tis, EMB[[emb]]$sfx)))
    stopifnot(nrow(stored) == sum(b$keep), max(abs(stored$p - f0$p)) < 1e-9)
    message(sprintf("%s / %s: reproduced %d neighbourhoods", tis, emb, sum(b$keep)))
    # ---- V1 symmetry audit (rotations and reflections of the ZT-ordered positions) and distinctness of the classes
    if (emb == "scanvi") {
      ord <- order(b$zt); zs <- b$zt[ord]
      syms <- c(lapply(0:5, function(r) (0:5 + r) %% 6), lapply(0:5, function(r) (-(0:5) + r) %% 6))
      dF <- vapply(syms, function(sy) {
        z <- b$zt; z[ord] <- zs[sy + 1]
        max(abs(fit_rhythm(b$ck, b$lib, z)$F - f0$F) / pmax(1, abs(f0$F)))
      }, numeric(1))
      des <- vapply(seq_along(classes), function(k) paste(round(class_times(b$zt, k)), collapse = ","), "")
      audit[[length(audit) + 1]] <- data.table(tissue = tis, check = c("max relative F change under the 12 symmetries",
                                                                       "distinct class designs"),
                                               value = c(max(dF), length(unique(des))), expected = c("< 1e-6", "60"))
    }
    # ---- V3 fits of all 60 classes (class 1 = observed)
    fits <- parallel::mclapply(seq_along(classes), function(k) {
      fk <- if (k == 1) f0 else fit_rhythm(b$ck, b$lib, class_times(b$zt, k))
      data.table(nhood = which(b$keep), relabel_class = k, F = fk$F, p = fk$p, peak_zt = fk$peak,
                 spatial_fdr = spatial_fdr(fk$p, b$w))
    }, mc.cores = NCORES)
    stopifnot(!vapply(fits, inherits, logical(1), "try-error"))
    fwrite(rbindlist(fits)[, `:=`(tissue = tis, embedding = emb)],
           file.path(out_dir, sprintf("milo_relabel_fits_%s_%s.csv", tis, emb)))
    if (emb != "scanvi") next
    # ---- membership of the rhythmic neighbourhoods (primary) for the external check
    sf0 <- spatial_fdr(f0$p, b$w)
    rh <- which(b$keep)[sf0 < 0.1]
    mem <- rbindlist(lapply(rh, function(j) data.table(nhood = j, cell_id = b$cells[b$nh[, j] > 0])))
    fwrite(mem[, tissue := tis], file.path(out_dir, sprintf("milo_membership_%s.csv", tis)))
    # ---- V2 positive control (primary embedding)
    set.seed(7)
    k0s <- sort(sample(2:60, 3))
    grid <- CJ(fc = c(1, 2, 3, 5, 8), k0 = k0s)
    for (g in seq_len(nrow(grid))) {
      fc <- grid$fc[g]; k0 <- grid$k0[g]
      set.seed(1000 + 10 * g)
      ck2 <- b$ck
      sel <- sample.int(nrow(ck2), round(0.05 * nrow(ck2)))
      if (fc > 1) {
        tk0 <- class_times(b$zt, k0)
        phi <- runif(length(sel), 0, 24)
        A <- log2(fc) / 2
        f <- 2^(A * cos(2 * pi / 24 * outer(phi, tk0, function(p, t) t - p)))
        f <- f / rowMaxs(f)
        ck2[sel, ] <- matrix(rbinom(length(f), as.vector(ck2[sel, ]), as.vector(f)), nrow = length(sel))
      }
      res <- parallel::mclapply(seq_along(classes), function(k) {
        fk <- fit_rhythm(ck2, b$lib, class_times(b$zt, k))
        list(p = fk$p, calls = sum(spatial_fdr(fk$p, b$w) < 0.1))
      }, mc.cores = NCORES)
      P <- vapply(res, function(r) r$p, numeric(nrow(ck2)))             # neighbourhoods x classes
      calls <- vapply(res, function(r) r$calls, numeric(1))
      best <- max.col(-P, ties.method = "random")
      nbest <- tabulate(best, nbins = 60)
      rk <- t(apply(-log10(P), 1, rank))                                    # higher = stronger rhythm
      rsum <- colSums(rk)
      stat_p <- function(v) mean(v >= v[k0])
      pc_rows[[length(pc_rows) + 1]] <- data.table(
        tissue = tis, fc = fc, k0 = k0, n_injected = if (fc > 1) length(sel) else 0L, n_tested = nrow(ck2),
        statistic = c("spatial-FDR < 0.1 calls", "best-class count", "rank sum of -log10 p"),
        observed = c(calls[k0], nbest[k0], rsum[k0]), null_mean = c(mean(calls[-k0]), mean(nbest[-k0]), mean(rsum[-k0])),
        null_q95 = c(quantile(calls[-k0], 0.95), quantile(nbest[-k0], 0.95), quantile(rsum[-k0], 0.95)),
        p_exact = c(stat_p(calls), stat_p(nbest), stat_p(rsum)),
        injected_best_is_k0 = if (fc > 1) mean(best[sel] == k0) else NA_real_)
      message(sprintf("PC %s fc %g k0 %d: p calls %.3f best %.3f ranksum %.3f", tis, fc, k0, stat_p(calls), stat_p(nbest),
                      stat_p(rsum)))
    }
  }
}
fwrite(rbindlist(audit), file.path(out_dir, "milo_audit_symmetry.csv"))
fwrite(rbindlist(pc_rows), file.path(out_dir, "milo_positive_control.csv"))
message("done")
