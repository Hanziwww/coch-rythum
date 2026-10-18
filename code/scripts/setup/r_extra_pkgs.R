# R packages not provided by conda (Module 0.1). Run WITHOUT the academic proxy: CRAN and Bioconductor come
# from the Tsinghua mirrors; GitHub packages are built from local clones made by github_clone.sh.
# Each install is attempted independently so one failure does not block the rest.
BIOC <- "https://mirrors.tuna.tsinghua.edu.cn/bioconductor/packages/3.22"
REPOS <- c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/",
           BioCsoft = file.path(BIOC, "bioc"), BioCann = file.path(BIOC, "data/annotation"),
           BioCexp = file.path(BIOC, "data/experiment"))
options(repos = REPOS, Ncpus = 8, timeout = 1800)
SRC <- file.path(Sys.getenv("COCH_ENVS", "envs"), "src")

try_install <- function(label, expr) {
  ok <- tryCatch({ eval(expr); TRUE }, error = function(e) { message("FAILED ", label, ": ", conditionMessage(e)); FALSE })
  ok <- ok && requireNamespace(label, quietly = TRUE)
  message(sprintf("[%s] %s", if (ok) "OK" else "FAIL", label))
  invisible(ok)
}

need <- function(pkgs) pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]

# CRAN + Bioconductor packages
for (p in need(c("limorhyde2", "circacompare", "nebula", "zeitzeiger", "deltaccd", "sccomp")))
  try_install(p, quote(install.packages(p)))

# dependencies of the cloned GitHub packages
deps_of <- function(dir) {
  d <- read.dcf(file.path(dir, "DESCRIPTION"))
  fields <- intersect(c("Depends", "Imports", "LinkingTo"), colnames(d))
  x <- unlist(strsplit(paste(d[1, fields], collapse = ","), ","))
  x <- trimws(gsub("\\(.*\\)", "", x))
  setdiff(x[nzchar(x)], c("R", rownames(installed.packages(priority = "base"))))
}
local_pkgs <- c(dryR = "dryR", diffCircadian = "diffCircadian", compareRhythms = "compareRhythms",
                nichenetr = "nichenetr", multinichenetr = "multinichenetr", InstaPrism = "InstaPrism",
                BayesPrism = "BayesPrism/BayesPrism", MuSiC = "MuSiC", tauFisher = "tauFisher")
dirs <- file.path(SRC, local_pkgs); names(dirs) <- names(local_pkgs)
dirs <- dirs[file.exists(file.path(dirs, "DESCRIPTION"))]
all_deps <- unique(unlist(lapply(dirs, deps_of)))
missing <- setdiff(need(all_deps), names(local_pkgs))
message("installing ", length(missing), " dependencies: ", paste(missing, collapse = ", "))
if (length(missing)) tryCatch(install.packages(missing), error = function(e) message("dep install error: ", conditionMessage(e)))

for (p in names(dirs)) if (!requireNamespace(p, quietly = TRUE))
  try_install(p, bquote(install.packages(.(dirs[[p]]), repos = NULL, type = "source")))

# CmdStan for sccomp (conda installs it under $CONDA_PREFIX/bin/cmdstan)
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  cand <- file.path(R.home(), "..", "..", "bin", "cmdstan")
  if (dir.exists(cand)) cmdstanr::set_cmdstan_path(normalizePath(cand))
  path <- tryCatch(cmdstanr::cmdstan_path(), error = function(e) NULL)
  message("cmdstan path: ", if (is.null(path)) "NOT FOUND" else path)
}

pk <- c("limma", "edgeR", "variancePartition", "mashr", "limorhyde2", "MetaCycle", "rain", "dryR",
        "circacompare", "diffCircadian", "compareRhythms", "nebula", "sccomp", "speckle", "UCell",
        "RcisTarget", "nichenetr", "multinichenetr", "InstaPrism", "BayesPrism", "MuSiC", "zeitzeiger",
        "tauFisher", "deltaccd", "tricycle", "msigdbr", "fgsea", "goseq", "qvalue", "glmmTMB", "circular")
st <- data.frame(package = pk,
                 installed = vapply(pk, requireNamespace, logical(1), quietly = TRUE),
                 version = vapply(pk, function(p) tryCatch(as.character(packageVersion(p)), error = function(e) NA_character_), character(1)))
print(st, row.names = FALSE)
cat("R_EXTRA_DONE\n")
