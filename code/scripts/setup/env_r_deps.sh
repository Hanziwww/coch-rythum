#!/usr/bin/env bash
# Second pass for the R env: compilers + prebuilt dependencies of packages that failed to build from source.
# One conda transaction; packages the solver reports as unavailable are dropped and the install is retried.
set -uo pipefail
ENV=${COCH_ENVS:-$COCH_ROOT/envs}/rcirc
CF=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
BC=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
CONDA=conda
unset http_proxy https_proxy

WANT="c-compiler cxx-compiler fortran-compiler make pkg-config
r-minpack.lm r-hmisc r-nmf r-ggpubr r-ggforce r-shadowtext r-tidygraph r-ggraph r-factoextra r-upsetr r-plotrix
r-combinat r-rmisc r-diagrammer r-mlrmbo r-parallelmap r-emoa r-dicekriging r-locfdr r-snowfall r-nnls r-mcmcpack
r-fda r-rcppparallel r-rfast r-dofuture r-dorng r-trust r-rcppeigen r-nloptr r-lme4 r-sysfonts r-gdtools
r-ash r-mclust r-pcapp r-leaps r-sparsem r-desolve r-zeallot r-hdinterval r-pbs r-dodr r-nebula
r-matrixstats bioconductor-scran bioconductor-sva bioconductor-toast bioconductor-sccomp bioconductor-scater
bioconductor-complexheatmap bioconductor-muscat bioconductor-milor"
WANT=$(echo $WANT)   # one line, single spaces

for attempt in 1 2 3 4 5 6; do
  echo "== attempt $attempt: $(echo $WANT | wc -w) packages"
  OUT=$($CONDA install -y -p "$ENV" --override-channels -c "$CF" -c "$BC" $WANT 2>&1)
  STATUS=$?
  echo "$OUT" | tail -25
  if [ $STATUS -eq 0 ]; then echo "R_DEPS_DONE"; exit 0; fi
  MISSING=$(echo "$OUT" | grep -oE "(nothing provides requested|  - )[a-z0-9._-]+" | grep -oE "[a-z0-9._-]+$" | sort -u)
  CONFLICT=$(echo "$OUT" | grep -oE "^[├└]─ [a-z0-9._-]+ .*(does not exist|is not installable)" | awk '{print $2}' | sort -u)
  DROP=$(echo -e "$MISSING\n$CONFLICT" | grep -v "^$" | sort -u)
  if [ -z "$DROP" ]; then echo "could not parse failure; stopping"; exit 1; fi
  echo "dropping: $DROP"
  for d in $DROP; do WANT=$(echo " $WANT " | sed "s/ $d / /g"); done
done
echo "R_DEPS_GAVE_UP"
