#!/usr/bin/env bash
# R environment for rhythm statistics, compositional models, CCC and deconvolution (Module 0.1).
# conda-forge + bioconda through the Tsinghua mirror; default (Anaconda) channels are not used.
set -euo pipefail

ENV=${COCH_ENVS:-$COCH_ROOT/envs}/rcirc
CF=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
BC=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
CONDA=conda

if [ ! -x "$ENV/bin/R" ]; then
  $CONDA create -y -p "$ENV" --override-channels -c "$CF" -c "$BC" --strict-channel-priority \
    "r-base=4.5" r-essentials r-data.table r-arrow r-remotes r-biocmanager r-matrix r-dplyr r-tidyr \
    r-circular r-glmmtmb r-metacycle r-mashr r-ashr r-msigdbr \
    r-cmdstanr cmdstan r-future r-furrr r-jsonlite r-yaml \
    bioconductor-limma bioconductor-edger bioconductor-deseq2 bioconductor-variancepartition \
    bioconductor-biocparallel bioconductor-rain bioconductor-speckle bioconductor-ucell \
    bioconductor-rcistarget bioconductor-tricycle bioconductor-qvalue bioconductor-fgsea \
    bioconductor-goseq bioconductor-singlecellexperiment bioconductor-scuttle \
    bioconductor-summarizedexperiment bioconductor-muscat
fi

echo "R_CONDA_DONE"
"$ENV/bin/Rscript" -e 'cat(R.version.string, "\n")'
