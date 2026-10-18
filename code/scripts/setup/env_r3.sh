#!/usr/bin/env bash
# Round 3: R packages for external single-cell matrices (cell calling, doublets). Adds new packages only (checked with
# --dry-run: no updates/downgrades of the existing rcirc environment). conda-forge + bioconda via the Tsinghua mirror.
set -uo pipefail
ENV=${COCH_ENVS:-$COCH_ROOT/envs}/rcirc
CF=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
BC=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
unset http_proxy https_proxy
conda install -y -p "$ENV" --override-channels -c "$CF" -c "$BC" \
  bioconductor-dropletutils bioconductor-scdblfinder && echo "R3_DEPS_DONE"
