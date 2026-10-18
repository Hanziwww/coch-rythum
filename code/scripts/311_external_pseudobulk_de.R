#!/usr/bin/env Rscript
# Round 3, WP2 — pseudobulk differential expression per transferred label in external noise datasets (limma-voom).
# Usage: Rscript scripts/311_external_pseudobulk_de.R <key>   (input from 310: cache/r3_query/<key>_pseudobulk.*)
# Per label: samples with >= 20 cells; contrasts need >= 2 samples in each group (gse346691: WT_noise vs WT_control;
# ma2025: day_noise vs control, night_noise vs control, night_noise vs day_noise; gse167078 has one pool per
# condition → no tests). edgeR filterByExpr + TMM, voom, lmFit ~ 0 + condition, eBayes(robust).
# Output (results/r3_<key>/): pseudobulk_de.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table) })
source("scripts/lib/cosinor.R")
key <- commandArgs(trailingOnly = TRUE)[1]
contr <- list(gse346691 = list(c("WT_noise", "WT_control")), gse167078 = list(),
              ma2025 = list(c("day_noise", "control"), c("night_noise", "control"), c("night_noise", "day_noise")))[[sub("ma2025dry", "ma2025", key)]]
out_dir <- file.path(RESULTS, paste0("r3_", key)); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
pb <- as.data.frame(arrow::read_parquet(file.path(CACHE, "r3_query", paste0(key, "_pseudobulk.parquet"))))
gcol <- intersect(c("__index_level_0__", "index"), names(pb))[1]
rownames(pb) <- pb[[gcol]]; pb[[gcol]] <- NULL
des <- fread(file.path(CACHE, "r3_query", paste0(key, "_pseudobulk_design.csv")))
res <- list()
for (lab in unique(des$label)) {
  d <- des[label == lab]
  for (cc in contr) {
    dd <- d[condition %in% cc]
    if (sum(dd$condition == cc[1]) < 2 || sum(dd$condition == cc[2]) < 2) next
    cnt <- as.matrix(pb[, dd$column]); storage.mode(cnt) <- "integer"
    grp <- factor(dd$condition, levels = cc)
    y <- DGEList(cnt, group = grp)
    keep <- filterByExpr(y, group = grp)
    if (sum(keep) < 100) next
    y <- calcNormFactors(y[keep, , keep.lib.sizes = FALSE])
    X <- model.matrix(~ 0 + grp); colnames(X) <- levels(grp)
    v <- voom(y, X)
    f <- eBayes(contrasts.fit(lmFit(v, X), makeContrasts(contrasts = paste0(cc[1], " - ", cc[2]), levels = X)), robust = TRUE)
    tt <- topTable(f, number = Inf, sort.by = "none")
    res[[length(res) + 1]] <- data.table(label = lab, contrast = paste0(cc[1], "_vs_", cc[2]), gene = rownames(tt),
                                         logFC = tt$logFC, AveExpr = tt$AveExpr, t = tt$t, p = tt$P.Value, q_BH = tt$adj.P.Val,
                                         n_a = sum(grp == cc[1]), n_b = sum(grp == cc[2]))
  }
}
res <- rbindlist(res)
if (!nrow(res)) {                                   # no contrast with >= 2 samples per group (e.g. one hashed pool each)
  unlink(file.path(out_dir, "pseudobulk_de.csv"))
  message("no testable contrast"); quit(save = "no")
}
fwrite(res, file.path(out_dir, "pseudobulk_de.csv"))
if (nrow(res)) print(res[, .(n_genes = .N, q05 = sum(q_BH < 0.05), p01 = sum(p < 0.01)), by = .(label, contrast)][order(-q05)][1:20])
message("done")
