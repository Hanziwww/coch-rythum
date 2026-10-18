#!/usr/bin/env Rscript
# Round 2, WP6 pilot — public treatment-response data for the cochlea (bulk RNA-seq).
# (1) GSE107086 (Cederroth 2019, PMID 31353184; CBA/Sca, 12 wk, whole cochlea, n = 3 per group): noise trauma
#     (6–12 kHz, 100 dB SPL, 1 h) in the day (ZT3) or night (ZT15) vs sham noise, crossed with adrenalectomy (ADX) vs
#     sham surgery. GEO lists collection at ZT11 (day) / ZT23 (night); the series design says 2 h after trauma — the
#     collection time is therefore inferred from the clock genes in the Python step (252). limma-voom on raw counts,
#     8-group design; contrasts: noise effect by time of day and surgery, their interactions, and the night−day
#     difference without noise.
# (2) GSE312476 (CBA/CaJ, 16 wk, whole cochlea, n = 3): mock vs 100 dB broadband noise for 2 h, sacrificed 0, 4 or
#     8 h after exposure; per-sample normalised values → limma-trend on log2(x + 1); each time vs mock.
# (3) Reference for predictions: tissue-level pseudobulk of our cochlea (all cell types summed per sample) — limma-voom
#     cosinor (ZT), so external day/night differences can be predicted from our fits.
# Outputs (results/r2_pilots/): pilot_GSE107086_de.csv, pilot_GSE312476_de.csv, cochlea_tissue_pseudobulk_cosinor.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "r2_pilots"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
geo <- file.path(ROOT, "external/geo")

# ---- (1) GSE107086 (file uses carriage-return line endings)
txt <- readLines(gzfile(file.path(geo, "GSE107086/GSE107086_raw_counts.csv.gz")), warn = FALSE)
txt <- unlist(strsplit(txt, "\r", fixed = TRUE))
cnt <- fread(text = paste(txt[nzchar(txt)], collapse = "\n"))
genes <- cnt$geneName
cnt <- as.matrix(cnt[, -1]); rownames(cnt) <- genes
cnt <- cnt[!duplicated(genes), ]
smp <- data.table(sample = colnames(cnt))
smp[, `:=`(noise = ifelse(grepl("^Sh[DN]N", sample), "sham_noise", "noise"),
           tod = ifelse(grepl("^(Sh)?DN", sample), "day", "night"),
           surgery = ifelse(grepl("ShADX", sample), "sham_surgery", "ADX"))]
smp[, group := factor(paste(noise, tod, surgery, sep = "."))]
stopifnot(all(table(smp$group) == 3))
X <- model.matrix(~ 0 + group, smp); colnames(X) <- levels(smp$group)
y <- DGEList(cnt)
keep <- filterByExpr(y, group = smp$group)
y <- calcNormFactors(y[keep, , keep.lib.sizes = FALSE])
v <- voom(y, X)
fit <- lmFit(v, X)
C <- makeContrasts(
  noise_day_shamsurg = noise.day.sham_surgery - sham_noise.day.sham_surgery,
  noise_night_shamsurg = noise.night.sham_surgery - sham_noise.night.sham_surgery,
  noise_night_vs_day_shamsurg = (noise.night.sham_surgery - sham_noise.night.sham_surgery) -
    (noise.day.sham_surgery - sham_noise.day.sham_surgery),
  noise_day_adx = noise.day.ADX - sham_noise.day.ADX,
  noise_night_adx = noise.night.ADX - sham_noise.night.ADX,
  noise_night_vs_day_adx = (noise.night.ADX - sham_noise.night.ADX) - (noise.day.ADX - sham_noise.day.ADX),
  three_way_gc_dependence = ((noise.night.sham_surgery - sham_noise.night.sham_surgery) -
    (noise.day.sham_surgery - sham_noise.day.sham_surgery)) - ((noise.night.ADX - sham_noise.night.ADX) -
    (noise.day.ADX - sham_noise.day.ADX)),
  night_minus_day_no_noise_shamsurg = sham_noise.night.sham_surgery - sham_noise.day.sham_surgery,
  night_minus_day_no_noise_adx = sham_noise.night.ADX - sham_noise.day.ADX,
  adx_effect_no_noise_day = sham_noise.day.ADX - sham_noise.day.sham_surgery,
  levels = X)
f2 <- eBayes(contrasts.fit(fit, C), robust = TRUE)
de1 <- rbindlist(lapply(colnames(C), function(k) {
  tt <- topTable(f2, coef = k, number = Inf, sort.by = "none")
  data.table(dataset = "GSE107086", contrast = k, gene = rownames(tt), logFC = tt$logFC, AveExpr = tt$AveExpr,
             t = tt$t, p = tt$P.Value, q_BH = tt$adj.P.Val)
}))
fwrite(de1, file.path(out_dir, "pilot_GSE107086_de.csv"))
print(de1[, .(n_q05 = sum(q_BH < 0.05), n_q10 = sum(q_BH < 0.1), n_p01 = sum(p < 0.01)), by = contrast])

# ---- (2) GSE312476
fs <- sort(list.files(file.path(geo, "GSE312476/raw"), pattern = "txt.gz$", full.names = TRUE))
# files carry 0–2 preamble lines, CRLF endings and a trailing tab: start at the header line
mats <- lapply(fs, function(f) { d <- fread(f, skip = "gene_symbol", fill = TRUE); setNames(as.numeric(d[[3]]), d$gene_symbol) })
gs <- Reduce(intersect, lapply(mats, names))
gs <- gs[!is.na(gs) & nzchar(gs)]
E <- sapply(mats, function(m) { m <- m[!duplicated(names(m))]; m[gs] })
colnames(E) <- sub("^GSM[0-9]+_", "", sub(".txt.gz$", "", basename(fs)))
grp <- factor(sub("_[0-9]+$", "", colnames(E)), levels = c("Cochlea_control", "PN0h", "PN4h", "PN8h"))
L <- log2(E + 1)
L <- L[rowMeans(L) > 2 & rowSums(E > 10) >= 3, ]
X2 <- model.matrix(~ grp)
f3 <- eBayes(lmFit(L, X2), trend = TRUE, robust = TRUE)
de2 <- rbindlist(lapply(2:4, function(k) {
  tt <- topTable(f3, coef = k, number = Inf, sort.by = "none")
  data.table(dataset = "GSE312476", contrast = paste0(levels(grp)[k], "_vs_mock"), gene = rownames(tt), logFC = tt$logFC,
             AveExpr = tt$AveExpr, t = tt$t, p = tt$P.Value, q_BH = tt$adj.P.Val)
}))
fwrite(de2, file.path(out_dir, "pilot_GSE312476_de.csv"))
print(de2[, .(n_q05 = sum(q_BH < 0.05), n_up = sum(q_BH < 0.05 & logFC > 0)), by = contrast])

# ---- (3) our cochlea, tissue-level pseudobulk cosinor
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
PB <- read_pseudobulk("cochlea")
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & tissue == "cochlea" & status != "excluded"]
tot <- NULL
for (g in incl$group) {
  gc <- group_counts(PB, design_dt, "cochlea", g)
  m <- gc$counts[, order(gc$zt)]
  tot <- if (is.null(tot)) m else tot + m[rownames(tot), ]
  zt <- sort(gc$zt)
}
tot <- tot[expr_filter(tot), ]
r <- limma_cosinor(tot, zt)$table
ap <- amp_phase(r$a, r$b)
r[, `:=`(peak_zt = ap$peak, amp_log2 = ap$amp, ptt_fc = 2^(2 * ap$amp))]
fwrite(r, file.path(out_dir, "cochlea_tissue_pseudobulk_cosinor.csv"))
message(sprintf("tissue pseudobulk: %d genes, %d with p < 0.05", nrow(r), sum(r$p < 0.05)))
message("done")
