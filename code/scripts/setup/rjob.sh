#!/usr/bin/env bash
# Run an R script with single-threaded BLAS/OpenMP (forked BiocParallel workers + OpenBLAS threads crash otherwise).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
exec ${COCH_RSCRIPT:-Rscript} "$@"
