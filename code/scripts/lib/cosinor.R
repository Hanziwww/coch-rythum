# Shared R helpers for 24-h rhythm analyses of 6-point pseudobulk time courses.
suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(yaml)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

# Scripts are run from the project root on the server (cd $COCH_ROOT).
CFG <- yaml::read_yaml(Sys.getenv("COCH_CONFIG", "scripts/config.yaml"))
ROOT <- Sys.getenv("COCH_ROOT", CFG$paths$root)
RESULTS <- file.path(ROOT, CFG$paths$results)
CACHE <- file.path(ROOT, CFG$paths$cache)
W24 <- 2 * pi / 24

cosinor_design <- function(zt, period = 24) {
  w <- 2 * pi / period
  cbind(Intercept = 1, cos = cos(w * zt), sin = sin(w * zt))
}

amp_phase <- function(a, b, period = 24) {
  list(amp = sqrt(a^2 + b^2), peak = (atan2(b, a) %% (2 * pi)) * period / (2 * pi))
}

circ_diff <- function(a, b, period = 24) {
  d <- (a - b) %% period
  ifelse(d > period / 2, d - period, d)
}

# Round 2: clock time alongside ZT, and delta-method CIs of the cosinor peak phase (hours).
zt_to_clock <- function(zt) (zt + (CFG$design$lights_on_clock_h %||% 8)) %% 24
phase_ci_delta <- function(a, b, var_a, var_b, cov_ab = 0, df = NULL, level = 0.95) {
  A2 <- a^2 + b^2
  var_phi <- (b^2 * var_a + a^2 * var_b - 2 * a * b * cov_ab) / pmax(A2^2, 1e-300)
  q <- if (is.null(df)) qnorm(0.5 + level / 2) else qt(0.5 + level / 2, df)
  list(peak = (atan2(b, a) %% (2 * pi)) / W24, half_width_h = pmin(q * sqrt(var_phi) / W24, 12))
}

circ_sd_h <- function(peaks_h) {            # rows = genes, cols = replicates/refits
  ang <- peaks_h * W24
  R <- sqrt(rowMeans(cos(ang))^2 + rowMeans(sin(ang))^2)
  sqrt(-2 * log(pmax(pmin(R, 1), 1e-12))) / W24
}

# Representatives of S_6 / D_6 (identity first): relabelings equivalent under rotation/reflection
# of the 6 equally spaced time labels give identical cosinor fits.
dihedral_classes <- function(n = 6) {
  perms <- function(v) {
    if (length(v) <= 1) return(list(v))
    out <- list()
    for (i in seq_along(v)) for (p in perms(v[-i])) out[[length(out) + 1]] <- c(v[i], p)
    out
  }
  all <- perms(0:(n - 1))
  key <- function(p) paste(p, collapse = "")
  group <- c(lapply(0:(n - 1), function(r) ((0:(n - 1)) + r) %% n),
             lapply(0:(n - 1), function(r) ((-(0:(n - 1))) + r) %% n))
  seen <- new.env(); reps <- list()
  for (p in all) {
    if (!is.null(seen[[key(p)]])) next
    orbit <- lapply(group, function(g) g[p + 1])
    for (o in orbit) assign(key(o), TRUE, envir = seen)
    keys <- vapply(orbit, key, "")
    reps[[length(reps) + 1]] <- orbit[[order(keys)[1]]]
  }
  ident <- 0:(n - 1)
  is_id <- vapply(reps, function(p) all(p == ident), TRUE)
  reps <- c(reps[is_id], reps[!is_id])
  if (n == 6) stopifnot(length(reps) == 60)
  stopifnot(all(reps[[1]] == ident))
  reps
}

read_pseudobulk <- function(tissue, level = "cell_type") {
  tab <- arrow::read_parquet(file.path(CACHE, "30_pseudobulk", sprintf("%s__%s.parquet", tissue, level)),
                             as_data_frame = TRUE)
  idx_col <- setdiff(names(tab), grep("|", names(tab), fixed = TRUE, value = TRUE))   # pandas index column
  stopifnot(length(idx_col) == 1)
  genes <- tab[[idx_col]]
  tab[[idx_col]] <- NULL
  m <- as.matrix(tab)
  storage.mode(m) <- "double"
  rownames(m) <- genes
  m
}

group_counts <- function(pb, design_dt, .tissue, .group, .level = "cell_type") {
  # argument names are dotted so data.table does not resolve them to the identically named columns
  d <- as.data.frame(design_dt)
  d <- d[d$tissue == .tissue & d$level == .level & d$group == .group, ]
  d <- d[order(d$zt), ]
  stopifnot(nrow(d) == 6)
  cols <- paste0(.group, "|", d$sample_id)
  list(counts = pb[, cols, drop = FALSE], zt = d$zt, sample_id = d$sample_id, n_cells = d$n_cells)
}

# Expression filter: CPM above the 10-count equivalent at the median library size in >= 3 of 6 samples.
expr_filter <- function(counts, min_count = 10, min_samples = 3) {
  lib <- colSums(counts)
  thr <- min_count / median(lib) * 1e6
  cpm <- t(t(counts) / lib) * 1e6
  rowSums(cpm >= thr) >= min_samples & rowSums(counts) >= 2 * min_count
}

# limma-voom cosinor fit; returns per-gene table (mesor, a, b, se, F, p) plus the fit object.
limma_cosinor <- function(counts, zt, norm_factors = NULL) {
  y <- edgeR::DGEList(counts)
  y <- if (is.null(norm_factors)) edgeR::calcNormFactors(y) else { y$samples$norm.factors <- norm_factors; y }
  X <- cosinor_design(zt)
  v <- limma::voom(y, X)
  fit <- limma::eBayes(limma::lmFit(v, X), robust = TRUE)
  tt <- limma::topTable(fit, coef = 2:3, number = Inf, sort.by = "none")
  se <- fit$stdev.unscaled * sqrt(fit$s2.post)
  list(table = data.table(gene = rownames(fit$coefficients), mesor = fit$coefficients[, 1],
                          a = fit$coefficients[, 2], b = fit$coefficients[, 3],
                          se_a = se[, 2], se_b = se[, 3], df_total = fit$df.total,
                          F = tt$F, p = tt$P.Value, q_BH = tt$adj.P.Val, AveExpr = tt$AveExpr),
       norm_factors = y$samples$norm.factors, fit = fit)
}

# Parametric bootstrap CIs for amplitude (log2) and peak phase from moderated SEs (t with df.total).
boot_ci <- function(a, b, se_a, se_b, df, nboot = 2000, seed = 1) {
  set.seed(seed)
  G <- length(a)
  ta <- matrix(rt(G * nboot, df = rep(df, nboot)), G, nboot)
  tb <- matrix(rt(G * nboot, df = rep(df, nboot)), G, nboot)
  A <- a + se_a * ta
  B <- b + se_b * tb
  amp <- sqrt(A^2 + B^2)
  pk <- (atan2(B, A) %% (2 * pi)) * 24 / (2 * pi)
  pk0 <- (atan2(b, a) %% (2 * pi)) * 24 / (2 * pi)
  dev <- circ_diff(pk, pk0)
  q <- if (requireNamespace("matrixStats", quietly = TRUE)) {
    function(m, p) matrixStats::rowQuantiles(m, probs = p)
  } else {
    function(m, p) apply(m, 1, quantile, probs = p, names = FALSE)
  }
  data.table(amp_ci_low = q(amp, 0.025), amp_ci_high = q(amp, 0.975),
             peak_ci_low = (pk0 + q(dev, 0.025)) %% 24, peak_ci_high = (pk0 + q(dev, 0.975)) %% 24,
             peak_ci_halfwidth_h = (q(dev, 0.975) - q(dev, 0.025)) / 2)
}
