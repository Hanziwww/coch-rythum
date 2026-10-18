#!/usr/bin/env bash
# Sequential, low-priority chain of analyses whose inputs exist (Modules 4–6).
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
P="nice -n 10 ${COCH_PYTHON:-python}"
export XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=8
for s in 80_atlas_catalogue 83_hearing_chronopharm 60_migration_models 65_migration_axes; do
  $P scripts/$s.py > logs/$s.log 2>&1; echo "$s exit $?" >> logs/chain_m456.log
done
nice -n 10 scripts/setup/rjob.sh scripts/51_abundance_models.R > logs/51_abundance_models.log 2>&1; echo "51 exit $?" >> logs/chain_m456.log
for s in 81_cell_cycle 66_state_continua 63_ccc_liana; do
  $P scripts/$s.py > logs/$s.log 2>&1; echo "$s exit $?" >> logs/chain_m456.log
done
echo CHAIN_DONE >> logs/chain_m456.log
