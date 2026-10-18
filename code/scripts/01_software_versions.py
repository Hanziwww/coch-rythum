"""Module 0.1 — software versions and compute environment used for all analyses.
Outputs (results/00_env/): software_versions.csv (language, package, version), compute_environment.csv
"""
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import ROOT, log, write_csv  # noqa: E402

# Every third-party package imported by the analysis scripts (round 9, Reporting Summary: completed with matplotlib,
# tensorly, DropletUtils, scuttle, BiocNeighbors, RcisTarget, deltaccd, SeuratObject, Matrix, nloptr and dplyr, and the
# --target overlay that 630 adds to sys.path for cellrank / palantir; Scrublet runs inside Scanpy).
PY = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "anndata", "scanpy", "scvi-tools", "scib-metrics",
      "harmonypy", "scanorama", "bbknn", "decoupler", "liana", "cell2cell", "numpyro", "jax", "torch", "lightning",
      "pyarrow", "joblib", "numba", "pynndescent", "leidenalg", "igraph", "pyyaml", "matplotlib", "tensorly"]
OVERLAY = Path("envs/traj_pkgs")
OVERLAY_PKGS = ["cellrank", "palantir", "mellon", "pygam", "scipy"]
R_PKGS = ["limma", "edgeR", "variancePartition", "mashr", "limorhyde2", "MetaCycle", "rain", "dryR", "circacompare",
          "diffCircadian", "sccomp", "cmdstanr", "speckle", "glmmTMB", "TMB", "BayesPrism", "miloR", "nichenetr",
          "multinichenetr", "SingleCellExperiment", "data.table", "arrow", "BiocParallel", "matrixStats",
          "DropletUtils", "scuttle", "BiocNeighbors", "RcisTarget", "deltaccd", "SeuratObject", "Matrix", "nloptr",
          "dplyr"]
rows = [{"language": "python", "package": "python", "version": platform.python_version()}]
for p in PY:
    try:
        rows.append({"language": "python", "package": p, "version": md.version(p)})
    except md.PackageNotFoundError:
        rows.append({"language": "python", "package": p, "version": "not installed"})
if OVERLAY.exists():
    found = {d.metadata["Name"].lower(): d.version for d in md.distributions(path=[str(OVERLAY)])}
    for p in OVERLAY_PKGS:
        rows.append({"language": "python overlay (630)", "package": p, "version": found.get(p, "not installed")})
rscript =Path("Rscript")
expr = ("cat('R', as.character(getRversion()), '\\n', sep='\\t'); for (p in c(" + ",".join(f"'{p}'" for p in R_PKGS)
        + ")) cat(p, tryCatch(as.character(packageVersion(p)), error = function(e) 'not installed'), '\\n', sep='\\t')")
out = subprocess.run([str(rscript), "-e", expr], capture_output=True, text=True, timeout=300).stdout
for line in out.strip().splitlines():
    p, v = (line.strip().split("\t") + [""])[:2]
    rows.append({"language": "R", "package": p, "version": v.strip()})
write_csv(pd.DataFrame(rows), "00_env", "software_versions.csv", "Software versions (Python env 'sc', R env 'rcirc')",
          required=["language", "package", "version"])


def read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return ""


q = read("/sys/fs/cgroup/cpu.max").split()
quota = f"{int(q[0]) / int(q[1]):.0f} cores" if len(q) == 2 and q[0] != "max" else "unlimited"
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
except Exception:  # torch optional here
    gpu = "unknown"
env = [("hostname", platform.node()), ("os", platform.platform()), ("visible_cpus", str(os.cpu_count())),
       ("cgroup_cpu_quota", quota), ("memory_gb", str(round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9))),
       ("gpu", gpu), ("project_root", str(ROOT))]
write_csv(pd.DataFrame(env, columns=["item", "value"]), "00_env", "compute_environment.csv",
          "Compute environment of the analysis server", required=["item", "value"])
log("done")
