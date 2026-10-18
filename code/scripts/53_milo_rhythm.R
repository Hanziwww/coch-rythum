#!/usr/bin/env Rscript
# Module 4 — neighbourhood-level rhythmic abundance (Milo; Dann 2022) on the selected per-tissue embeddings.
# Re-implemented with sparse operations (miloR 2.6 makeNhoods runs a kNN search with k = number of sampled
# neighbourhoods and fills matrices element-wise; countCells loops over cells — hours on these graphs):
#   kNN (k = 30) on the selected latent space (results/20_dr/embedding_<tissue>_latent.csv);
#   refined sampling: random prop of cells (0.05 cochlea / 0.1 BM) → median latent position of each one's 30 NN →
#     nearest cell = index cell (duplicates dropped); neighbourhood = index cell + its 30 nearest neighbours;
#   counts per neighbourhood × sample = crossprod(neighbourhood matrix, sample indicator);
#   edgeR quasi-likelihood NB GLM ~ cos + sin (6 samples), 2-df F test of the rhythm;
#   spatial FDR = weighted BH with k-distance weights (1 / distance of the index cell to its 30th neighbour), as in
#     miloR::graphSpatialFDR(weighting = "k-distance").
# Neighbourhoods are annotated with the majority gold-standard cell type and its fraction (sub-states, not labels).
# Outputs (results/53_milo/): milo_nhoods_<tissue>.csv, milo_da_results_<tissue>.csv
suppressPackageStartupMessages({ library(BiocNeighbors); library(edgeR); library(data.table); library(Matrix)
  library(matrixStats) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "53_milo"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
K <- 30
TISSUES <- strsplit(Sys.getenv("TISSUES", "bm,cochlea"), ",")[[1]]
LAT <- Sys.getenv("LATENT_TEMPLATE", file.path(RESULTS, "20_dr/embedding_%s_latent.csv"))   # sensitivity: other latent
SFX <- Sys.getenv("OUT_SUFFIX", "")
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
  smp <- unique(obs[, .(sample_id, zt)])[match(colnames(cnt), sample_id)]
  X <- cosinor_design(smp$zt)
  keep <- rowSums(cnt) >= 10
  y <- DGEList(cnt[keep, ], lib.size = colSums(cnt))
  y <- estimateDisp(y, X)
  qt <- glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)
  co <- qt$coefficients
  ap <- amp_phase(co[, 2], co[, 3])
  p <- qt$table$PValue
  w <- 1 / knn$distance[idx[keep], K]
  o <- order(p)
  sfdr <- numeric(length(p))
  sfdr[o] <- pmin(1, rev(cummin(rev(sum(w) * p[o] / cumsum(w[o])))))
  labs <- sort(unique(obs$cell_type))
  L <- sparseMatrix(i = seq_len(n), j = match(obs$cell_type, labs), x = 1, dims = c(n, length(labs)))
  lc <- as.matrix(crossprod(nh, L))
  size <- rowSums(lc)
  maj_lab <- labs[max.col(lc, ties.method = "first")]
  maj_frac <- apply(lc, 1, max) / size
  fwrite(data.table(tissue = tis, nhood = seq_along(idx), index_cell = rownames(Z)[idx], size = size,
                    majority_cell_type = maj_lab, majority_fraction = maj_frac, k_distance = knn$distance[idx, K],
                    tested = keep),
         file.path(out_dir, sprintf("milo_nhoods_%s%s.csv", tis, SFX)))
  res <- data.table(tissue = tis, nhood = which(keep), majority_cell_type = maj_lab[keep],
                    majority_fraction = maj_frac[keep], mean_count = rowMeans(cnt[keep, ]),
                    logFC_cos = co[, 2] / log(2), logFC_sin = co[, 3] / log(2), amp_log2 = ap$amp / log(2),
                    ptt_fc = exp(2 * ap$amp), peak_zt = ap$peak, F = qt$table$F, p = p, spatial_fdr = sfdr)
  fwrite(res, file.path(out_dir, sprintf("milo_da_results_%s%s.csv", tis, SFX)))
  message(sprintf("%s: %d neighbourhoods (%d tested); spatial FDR < 0.1: %d", tis, length(idx), nrow(res),
                  sum(res$spatial_fdr < 0.1)))
  print(res[spatial_fdr < 0.2, .(n = .N, median_peak_zt = median(peak_zt)), by = majority_cell_type][order(-n)])
}
message("done")
