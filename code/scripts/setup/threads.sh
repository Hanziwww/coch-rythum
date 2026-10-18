# Source to cap thread pools: the container's CPU quota is 24 cores (cgroup cpu.max) although nproc reports 208,
# and BLAS/OpenMP/numba/torch otherwise start 208 threads each and get throttled.
T=${THREADS:-4}
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMBA_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T \
       VECLIB_MAXIMUM_THREADS=$T RAYON_NUM_THREADS=$T POLARS_MAX_THREADS=$T
