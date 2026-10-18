#!/usr/bin/env Rscript
# Module 2.6 — differential rhythmicity between the independent cochlea and temporal-BM cohorts for cell types
# included (status 'primary') in both tissues. Model selection with dryR::dryseq (NB GLM; tissue-specific means
# absorb the platform difference), 5 models for 2 groups:
#   1 arrhythmic in both | 2 rhythmic in BM only | 3 rhythmic in cochlea only | 4 shared rhythm | 5 different rhythms
# (model numbering/labels are taken from dryR output and mapped explicitly below), with BIC weights.
# For genes with a rhythmic model (BICW of chosen model >= 0.5), CircaCompare estimates Δmesor/Δamplitude/Δphase
# on within-tissue log2-CPM (Δmesor reflects platform and is not interpreted).
# Usage: Rscript scripts/35_crosstissue_dryR.R
suppressPackageStartupMessages({
  library(edgeR); library(data.table); library(BiocParallel); library(dryR); library(circacompare)
})
source("scripts/lib/cosinor.R")
NCORES <- as.integer(Sys.getenv("NCORES", "8"))   # inner cores for dryseq/DESeq2 (cell types run serially)
out_dir <- file.path(RESULTS, "35_crosstissue"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type"]
shared <- intersect(incl[tissue == "cochlea" & status == "primary", group], incl[tissue == "bm" & status == "primary", group])
message("shared cell types: ", length(shared), " — ", paste(shared, collapse = "; "))
PB <- list(cochlea = read_pseudobulk("cochlea"), bm = read_pseudobulk("bm"))

cache_dir35 <- file.path(CACHE, "35_crosstissue"); dir.create(cache_dir35, recursive = TRUE, showWarnings = FALSE)
run_type <- function(ct) {
  cf <- file.path(cache_dir35, paste0(gsub("[^A-Za-z0-9]+", "_", ct), ".rds"))
  if (file.exists(cf)) return(readRDS(cf))
  gcc <- group_counts(PB$cochlea, design_dt, "cochlea", ct)
  gbm <- group_counts(PB$bm, design_dt, "bm", ct)
  keep <- expr_filter(gcc$counts) & expr_filter(gbm$counts)
  cnt <- cbind(gcc$counts[keep, ], gbm$counts[keep, ])
  storage.mode(cnt) <- "integer"
  grp <- rep(c("cochlea", "bm"), each = 6)
  tt <- c(gcc$zt, gbm$zt)
  colnames(cnt) <- paste0(grp, "_ZT", tt)
  # dryseq calls DESeq(parallel = TRUE), which uses BiocParallel's registered backend (default = all cores):
  # register a bounded backend explicitly to avoid nested worker explosions.
  BiocParallel::register(BiocParallel::MulticoreParam(NCORES))
  dr <- dryseq(cnt, group = grp, time = tt, period = 24, sample_name = colnames(cnt), n.cores = NCORES)
  res <- as.data.table(dr$results, keep.rownames = "gene")
  res[, cell_type := ct]
  # CircaCompare on rhythmic genes (within-tissue log2-CPM)
  lc <- cbind(cpm(calcNormFactors(DGEList(gcc$counts[keep, ])), log = TRUE, prior.count = 1),
              cpm(calcNormFactors(DGEList(gbm$counts[keep, ])), log = TRUE, prior.count = 1))
  rhy_models <- unique(res$chosen_model)
  cand <- res[chosen_model != 1 & chosen_model_BICW >= 0.5, gene]
  cc <- rbindlist(parallel::mclapply(cand, mc.cores = NCORES, FUN = function(g) {
    df <- data.frame(time = tt, measure = as.numeric(lc[g, ]), group = grp)
    out <- tryCatch(circacompare(x = df, col_time = "time", col_group = "group", col_outcome = "measure",
                                 period = 24, alpha_threshold = 1)$summary, error = function(e) NULL)
    if (is.null(out)) return(NULL)
    w <- data.table(gene = g, parameter = out$parameter, value = out$value)
    dcast(w, gene ~ parameter, value.var = "value")
  }), fill = TRUE)
  if (nrow(cc)) cc[, cell_type := ct]
  message(sprintf("  %-35s genes=%5d  rhythmic-model genes=%4d", ct, nrow(res), length(cand)))
  out <- list(dry = res, cc = cc)
  saveRDS(out, cf)
  out
}

out <- lapply(shared, function(ct) tryCatch(run_type(ct), error = function(e) e))
failed <- vapply(out, function(x) inherits(x, "error"), TRUE)
if (any(failed)) message("FAILED: ", paste(shared[failed], collapse = "; "), "\n",
                         paste(vapply(out[failed], conditionMessage, ""), collapse = "\n"))
dry <- rbindlist(lapply(out[!failed], `[[`, "dry"), fill = TRUE)
cc <- rbindlist(lapply(out[!failed], `[[`, "cc"), fill = TRUE)
fwrite(dry, file.path(out_dir, "crosstissue_dryR.csv"))
fwrite(cc, file.path(out_dir, "crosstissue_circacompare.csv"))
summ <- dry[, .N, by = .(cell_type, chosen_model)][order(cell_type, chosen_model)]
summ_w <- dcast(summ, cell_type ~ paste0("model_", chosen_model), value.var = "N", fill = 0)
fwrite(summ_w, file.path(out_dir, "crosstissue_dryR_model_counts.csv"))
print(summ_w)
message("dryR model columns: ", paste(grep("BICW|model", names(dry), value = TRUE), collapse = ", "))
message("done")
