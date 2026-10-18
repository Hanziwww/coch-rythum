#!/usr/bin/env Rscript
# Module 2.8 — clock network per tissue × cell type.
# Clock correlation distance (CCD; Shilts et al. 2018, deltaccd): distance between the cell type's clock-gene
# co-expression across its 6 ZT samples and the mouse pan-tissue reference correlation pattern (Arntl absent → the
# reference is restricted to available genes); permutation p from shuffling gene labels (nPerm = 1000).
# Combined with the core-clock fits of 13_clock_controls (phase offset, coherence) into one summary table.
# Outputs (results/34_clock_network/): clock_ccd.csv, clock_network_by_celltype.csv
suppressPackageStartupMessages({ library(edgeR); library(data.table); library(deltaccd) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "34_clock_network"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status != "excluded"]
ref <- getRefCor(species = "mouse", tissue = "pan", useEntrezGeneId = FALSE)
# CCD computed directly (deltaccd's formula: Euclidean distance between the upper triangles of the Spearman
# correlation matrix of clock genes and the reference), robust to genes with zero variance in a group.
# Null: each gene's values permuted independently across the 6 samples (destroys co-expression), 1000 draws;
# p = P(CCD_null <= CCD_obs) (small distance = clock-like co-expression).
ccd_fun <- function(x, r) {                         # x: genes × samples
  cc <- suppressWarnings(cor(t(x), method = "spearman"))
  ut <- upper.tri(cc)
  ok <- !is.na(cc[ut])
  sqrt(sum((cc[ut][ok] - r[ut][ok])^2)) / sqrt(sum(ok)) * sqrt(sum(ut))
}
set.seed(1)
rows <- list()
for (tis in c("cochlea", "bm")) {
  pb <- read_pseudobulk(tis)
  for (g in incl[tissue == tis, group]) {
    gc <- group_counts(pb, design_dt, tis, g)
    lc <- cpm(calcNormFactors(DGEList(gc$counts)), log = TRUE, prior.count = 1)
    genes <- intersect(rownames(ref), rownames(lc))
    genes <- genes[apply(lc[genes, , drop = FALSE], 1, sd) > 0]
    if (length(genes) < 6) next
    x <- lc[genes, ]
    r <- ref[genes, genes]
    obs <- ccd_fun(x, r)
    null <- replicate(1000, ccd_fun(t(apply(x, 1, sample)), r))
    rows[[paste(tis, g)]] <- data.table(tissue = tis, group = g, CCD = obs, null_mean = mean(null),
                                        z = (obs - mean(null)) / sd(null), Pvalue = (1 + sum(null <= obs)) / 1001,
                                        n_clock_genes = length(genes), genes_used = paste(genes, collapse = ";"))
  }
}
ccd <- rbindlist(rows, fill = TRUE)
fwrite(ccd, file.path(out_dir, "clock_ccd.csv"))
order <- fread(file.path(RESULTS, "02_clock_controls/clock_phase_order.csv"))
setnames(ccd, "group", "cell_type", skip_absent = TRUE)
net <- merge(order, ccd, by = c("tissue", "cell_type"), all = TRUE)
fwrite(net, file.path(out_dir, "clock_network_by_celltype.csv"))
print(net[, .(tissue, cell_type, clock_phase_offset_h, phase_coherence, CCD = get(intersect(c("CCD", "ccd"), names(net))[1]),
              Pvalue = get(intersect(c("Pvalue", "pvalue", "p"), names(net))[1]))][order(tissue, -phase_coherence)])
message("done")
