#!/usr/bin/env Rscript
# Verification — simulation of the rhythm pipeline on the dataset's own noise.
# For 4 cell types per tissue spanning the depth range, the 6 pseudobulk samples are relabelled with a random
# non-identity dihedral class (true rhythms become non-sinusoidal 'noise' while all sample-level structure is kept).
# In 5 % of expressed genes (random) a 24-h rhythm with peak-to-trough fold change FC ∈ {1.25, 1.5, 2, 3} and uniform
# random phase is injected by binomial thinning of each sample's counts with f_s = 2^{A cos(ω(t_s − φ))} / max_s(·),
# A = log2(FC)/2 (thinning keeps the count noise structure). The pipeline is then re-run on the simulated data:
#   primary limma-voom cosinor (BH q), pooled-null permutation FDR over the 59 relabelings (as in 31), LOTO phase SD,
#   edgeR-QL (one independent method), and a Tier-1-like rule: q_BH < 0.05 & q_perm < 0.1 & FC >= 1.5 &
#   edgeR p < 0.05 & LOTO phase SD < 3 h (the single-sample criterion and RAIN/JTK are not emulated).
# Scores per rule: power among injected genes, false-discovery proportion among calls (non-injected genes are not
# strictly null — they keep relabelled biological and sample-level structure, which is the relevant alternative here),
# amplitude ratio (estimated/injected) and absolute phase error of detected injected genes, and the KS distance of
# non-injected p-values from uniform.
# Output (results/33_rhythm_tiers/): rhythm_pipeline_simulation.csv
suppressPackageStartupMessages({ library(limma); library(edgeR); library(data.table); library(parallel) })
source("scripts/lib/cosinor.R")
NCORES <- as.integer(Sys.getenv("NCORES", "6"))
design_dt <- fread(file.path(RESULTS, "30_pseudobulk/pseudobulk_design.csv"))
incl <- fread(file.path(RESULTS, "30_pseudobulk/groups_included.csv"))[level == "cell_type" & status == "primary"]
classes <- dihedral_classes(6)

perm_fdr <- function(cnt, zt, nf, F) {
  nullF <- sapply(classes[-1], function(cl) limma_cosinor(cnt, zt[cl + 1], norm_factors = nf)$table$F)
  pooled <- sort(as.vector(nullF))
  n_ge <- length(pooled) - findInterval(F, pooled, left.open = TRUE)
  o <- order(F, decreasing = TRUE)
  fdr <- pmin(1, (n_ge[o] / (length(classes) - 1)) / seq_along(o))
  q <- numeric(length(F)); q[o] <- rev(cummin(rev(fdr)))
  q
}

sim_one <- function(job) {
  set.seed(job$seed)
  cnt0 <- job$cnt0; zt <- job$zt
  cnt <- cnt0
  inj <- sample(nrow(cnt), round(0.05 * nrow(cnt)))
  phi <- runif(length(inj), 0, 24)
  A <- log2(job$fc) / 2
  f <- 2^(A * cos(W24 * outer(phi, zt, function(p, t) t - p)))
  f <- f / apply(f, 1, max)
  cnt[inj, ] <- matrix(rbinom(length(f), as.vector(cnt[inj, ]), as.vector(f)), nrow = length(inj))
  fit <- limma_cosinor(cnt, zt)
  res <- fit$table
  ap <- amp_phase(res$a, res$b)
  qperm <- perm_fdr(cnt, zt, fit$norm_factors, res$F)
  lo_peak <- sapply(seq_along(zt), function(j) {
    r <- limma_cosinor(cnt[, -j], zt[-j], norm_factors = fit$norm_factors[-j])$table
    amp_phase(r$a, r$b)$peak
  })
  loto_sd <- circ_sd_h(lo_peak)
  y <- DGEList(cnt); y$samples$norm.factors <- fit$norm_factors
  X <- cosinor_design(zt)
  y <- estimateDisp(y, X)
  ep <- glmQLFTest(glmQLFit(y, X, robust = TRUE), coef = 2:3)$table$PValue
  is_inj <- seq_len(nrow(res)) %in% inj
  rules <- list(q_BH_05 = res$q_BH < 0.05, q_BH_10 = res$q_BH < 0.1,
                q_BH_05_qperm_10 = res$q_BH < 0.05 & qperm < 0.1,
                tier1_like = res$q_BH < 0.05 & qperm < 0.1 & 2^(2 * ap$amp) >= 1.5 & ep < 0.05 & loto_sd < 3)
  rbindlist(lapply(names(rules), function(rn) {
    call <- rules[[rn]]; call[is.na(call)] <- FALSE
    det <- is_inj & call
    data.table(tissue = job$tissue, cell_type = job$cell_type, lib_total = sum(cnt0), replicate = job$rep,
               relabel_class = job$cls, ptt_fc_injected = job$fc, rule = rn, n_genes = nrow(res),
               n_injected = length(inj), n_calls = sum(call), power = mean(call[is_inj]),
               fdp = if (any(call)) mean(!is_inj[call]) else 0,
               median_amp_ratio = if (any(det)) median(ap$amp[det] / A) else NA_real_,
               median_abs_phase_error_h = if (any(det)) median(abs(circ_diff(ap$peak[det], phi[match(which(det), inj)])))
                                          else NA_real_,
               ks_null_D = unname(suppressWarnings(ks.test(res$p[!is_inj], "punif")$statistic)))
  }))
}

jobs <- list()
for (tis in c("bm", "cochlea")) {
  pb <- read_pseudobulk(tis)
  groups <- incl[tissue == tis, group]
  libs <- sapply(groups, function(g) sum(group_counts(pb, design_dt, tis, g)$counts))
  pick <- names(sort(libs))[unique(round(quantile(seq_along(libs), c(0.1, 0.4, 0.7, 1))))]
  for (g in pick) {
    gc <- group_counts(pb, design_dt, tis, g)
    cnt0 <- gc$counts[expr_filter(gc$counts), , drop = FALSE]
    for (rep in 1:3) {
      set.seed(100 * rep + match(g, groups))
      k <- sample(2:length(classes), 1)
      for (fc in c(1.25, 1.5, 2, 3))
        jobs[[length(jobs) + 1]] <- list(tissue = tis, cell_type = g, cnt0 = cnt0, zt = gc$zt[classes[[k]] + 1],
                                         cls = paste(classes[[k]], collapse = ""), rep = rep, fc = fc,
                                         seed = 1e4 * rep + 100 * fc + match(g, groups))
    }
  }
}
message(length(jobs), " simulated datasets")
sim <- rbindlist(mclapply(jobs, sim_one, mc.cores = NCORES))
fwrite(sim, file.path(RESULTS, "33_rhythm_tiers/rhythm_pipeline_simulation.csv"))
print(sim[, .(power = round(mean(power), 3), fdp = round(mean(fdp), 3), calls = round(mean(n_calls), 1),
              amp_ratio = round(median(median_amp_ratio, na.rm = TRUE), 2),
              phase_err = round(median(median_abs_phase_error_h, na.rm = TRUE), 2)),
          by = .(tissue, rule, ptt_fc_injected)][order(tissue, rule, ptt_fc_injected)])
message("done")
