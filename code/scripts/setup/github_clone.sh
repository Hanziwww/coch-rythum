#!/usr/bin/env bash
# Shallow-clone GitHub R packages through the academic proxy (github.com is on the accelerated list),
# then switch the proxy off again. Packages are built offline by r_extra_pkgs.R.
set -uo pipefail
SRC=${COCH_ENVS:-$COCH_ROOT/envs}/src
mkdir -p "$SRC"
: # site-specific network proxy (not needed elsewhere)
export GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt
for repo in naef-lab/dryR diffCircadian/diffCircadian bharathananth/compareRhythms saeyslab/nichenetr \
            saeyslab/multinichenetr humengying0907/InstaPrism Danko-Lab/BayesPrism xuranw/MuSiC micnngo/tauFisher; do
  name=${repo#*/}
  if [ -d "$SRC/$name/.git" ]; then echo "exists $name"; continue; fi
  if git clone --depth 1 "https://github.com/$repo.git" "$SRC/$name" >/dev/null 2>&1; then
    echo "cloned $repo"
  else
    echo "FAILED $repo"
  fi
done
unset http_proxy https_proxy
ls "$SRC"
