#!/usr/bin/env bash
# Python environment for the circadian atlas analyses (Module 0.1).
# venv with --system-site-packages so the base conda torch (2.12+cu130, RTX 5090 / sm_120) is reused.
set -euo pipefail

ENV=${COCH_ENVS:-$COCH_ROOT/envs}/sc
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

if [ ! -x "$ENV/bin/python" ]; then
  python -m venv --system-site-packages "$ENV"
fi
source "$ENV/bin/activate"
pip install -q -i "$PIP_INDEX" --upgrade pip

# Core single-cell stack + statistics. "only-if-needed" keeps the base torch build.
pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed \
  "scanpy>=1.11" anndata scikit-misc igraph leidenalg pyyaml pyarrow statsmodels scikit-learn \
  harmonypy scanorama bbknn "scvi-tools>=1.3" "scib-metrics>=0.5" \
  decoupler omnipath liana mudata cell2cell tensorly \
  numpyro "jax[cpu]" arviz pydeseq2 POT

python - <<'EOF'
import importlib, torch
mods = ["scanpy", "anndata", "scvi", "scib_metrics", "harmonypy", "scanorama", "bbknn",
        "decoupler", "liana", "cell2cell", "numpyro", "jax", "skmisc"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m:14s}", getattr(mod, "__version__", "ok"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
echo "PYTHON_ENV_DONE"
