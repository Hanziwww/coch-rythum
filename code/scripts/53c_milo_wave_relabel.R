#!/usr/bin/env Rscript
# Round 9 (D75, O15) — exact time-relabelling null for the Milo production waves (Fig. 5d; M1 / M3 / B02 method lines).
# The waves correlate each neighbourhood's peak ZT with its index cell's pseudotime on the unsupervised scVI
# neighbourhoods of 53_milo_rhythm.R (LATENT_TEMPLATE = cache/r2_embedding/latent_%s_scvi.csv, OUT_SUFFIX = _scvi).
# Their original null shuffled pseudotime across neighbourhoods, which treats thousands of neighbourhoods built from six
# samples as independent. Here the neighbourhoods are rebuilt with the same code and seed (stored statistics asserted)
# and the neighbourhood rhythm is refitted for each of the 60 dihedral relabelling classes of the sampling times
# (class 1 = observed); 53d_milo_wave_null.py recomputes the wave statistic per class.
# Output (results/53_milo/): milo_relabel_peaks_<tissue>_scvi.csv (nhood x relabel class: peak ZT, p)
suppressPackageStartupMessages({ library(BiocNeighbors); library(edgeR); library(data.table); library(Matrix)
  library(matrixStats) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "53_milo")
K <- 30
TISSUES <- strsplit(Sys.getenv("TISSUES", "bm,cochlea"), ",")[[1]]
LAT <- file.path(CACHE, "r2_embedding/latent_%s_scvi.csv")
NCORES <- as.integer(Sys.getenv("NCORES", "8"))
classes <- dihedral_classes(6)
stopifnot(length(classes) == 60, identical(as.integer(classes[[1]]), 0:5))

fit_rhythm <- function(cnt, lib, zt) {                # the test of 53_milo_rhythm.R
  X <- cosinor_design(zt)
  y <- DGEList(cnt, lib.size = lib)
  y <- estimateDisp(y, X)
  qt <- glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)
  co <- qt$coefficients
  list(F = qt$table$F, p = qt$table$PValue, peak = amp_phase(co[, 2], co[, 3])$peak)
}

for (tis in TISSUES) {
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
  zt <- unique(obs[, .(sample_id, zt)])[match(colnames(cnt), sample_id)]$zt
  keep <- rowSums(cnt) >= 10
  lib <- colSums(cnt)
  ck <- cnt[keep, , drop = FALSE]
  f0 <- fit_rhythm(ck, lib, zt)
  stored <- fread(file.path(out_dir, sprintf("milo_da_results_%s_scvi.csv", tis)))
  stopifnot(nrow(stored) == sum(keep), all(stored$nhood == which(keep)), max(abs(stored$p - f0$p)) < 1e-9,
            max(abs(((stored$peak_zt - f0$peak + 12) %% 24) - 12)) < 1e-6)
  message(sprintf("%s (scVI): reproduced %d tested neighbourhoods", tis, sum(keep)))
  ord <- order(zt); zs <- zt[ord]
  one_class <- function(k) {
    ztk <- zt
    ztk[ord] <- zs[classes[[k]] + 1]
    fk <- if (k == 1) f0 else fit_rhythm(ck, lib, ztk)
    data.table(nhood = which(keep), relabel_class = k, peak_zt = fk$peak, p = fk$p)
  }
  res <- parallel::mclapply(seq_along(classes), one_class, mc.cores = NCORES)
  stopifnot(!vapply(res, inherits, logical(1), "try-error"))
  fwrite(rbindlist(res)[, tissue := tis], file.path(out_dir, sprintf("milo_relabel_peaks_%s_scvi.csv", tis)))
}
message("done")
