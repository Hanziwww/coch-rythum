#!/usr/bin/env Rscript
# Round 2, WP5 — permutation check of the cross-tissue dryR model classes (35): dryR (dryseq, as in 35) re-run for
# three shared immune cell types with the cochlear time labels relabelled by 3 random non-identity dihedral classes
# (BM labels kept). Under these nulls any cochlear rhythm is spurious, so the number of genes assigned to
# cochlea_only / shared / different (BIC weight >= 0.5) estimates how many observed calls BIC model selection alone
# would produce from non-circadian structure. Output (results/r2_audit/): dryr_null_check.csv
suppressPackageStartupMessages({ library(edgeR); library(data.table); library(BiocParallel); library(dryR) })
source("scripts/lib/cosinor.R")
NC <- as.integer(Sys.getenv("NCORES", "4"))
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
PB <- list(cochlea = read_pseudobulk("cochlea"), bm = read_pseudobulk("bm"))
classes <- dihedral_classes(6)
obs <- fread(file.path(RESULTS, "35_crosstissue/crosstissue_dryR.csv"))
amp_c <- grep("^amp_.*cochlea", names(obs), value = TRUE)[1]
amp_b <- grep("^amp_.*bm", names(obs), value = TRUE)[1]
label_models <- function(d) {
  lab <- d[, .(z_c = mean(is.na(get(amp_c)) | abs(get(amp_c)) < 1e-12), z_b = mean(is.na(get(amp_b)) | abs(get(amp_b)) < 1e-12),
               same = mean(abs(get(amp_c) - get(amp_b)) < 1e-9, na.rm = TRUE)), by = chosen_model]
  lab[, label := fifelse(z_c > 0.5 & z_b > 0.5, "arrhythmic", fifelse(z_c > 0.5, "bm_only",
                  fifelse(z_b > 0.5, "cochlea_only", fifelse(same > 0.5, "shared", "different"))))]
  merge(d, lab[, .(chosen_model, model_label = label)], by = "chosen_model")
}
obs <- label_models(obs)
cts <- c("Neutrophil (Band)", "Monocyte (Classical)", "B Cell (Naïve)")
set.seed(11)
rows <- list()
BiocParallel::register(BiocParallel::MulticoreParam(NC))
for (ct in cts) {
  o <- obs[cell_type == ct & chosen_model_BICW >= 0.5, .N, by = model_label]
  rows[[length(rows) + 1]] <- data.table(cell_type = ct, run = "observed", relabel_class = "identity", o)
  gcc <- group_counts(PB$cochlea, design_dt, "cochlea", ct)
  gbm <- group_counts(PB$bm, design_dt, "bm", ct)
  keep <- expr_filter(gcc$counts) & expr_filter(gbm$counts)
  cnt <- cbind(gcc$counts[keep, ], gbm$counts[keep, ]); storage.mode(cnt) <- "integer"
  grp <- rep(c("cochlea", "bm"), each = 6)
  for (k in sample(2:length(classes), 3)) {
    tt <- c(gcc$zt[classes[[k]] + 1], gbm$zt)
    colnames(cnt) <- paste0(grp, "_ZT", tt, "_", seq_along(tt))
    dr <- dryseq(cnt, group = grp, time = tt, period = 24, sample_name = colnames(cnt), n.cores = NC)
    d <- label_models(as.data.table(dr$results, keep.rownames = "gene"))
    rows[[length(rows) + 1]] <- data.table(cell_type = ct, run = "null", relabel_class = paste(classes[[k]], collapse = ""),
                                           d[chosen_model_BICW >= 0.5, .N, by = model_label])
    message(ct, " null class ", paste(classes[[k]], collapse = ""), " done")
  }
}
res <- rbindlist(rows, fill = TRUE)
out_dir <- file.path(RESULTS, "r2_audit"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(res, file.path(out_dir, "dryr_null_check.csv"))
print(dcast(res, cell_type + run + relabel_class ~ model_label, value.var = "N", fill = 0))
message("done")
