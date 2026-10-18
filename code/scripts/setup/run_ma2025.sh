#!/usr/bin/env bash
# Ma 2025 chain (O01): prepare the user's upload, then 300 → 301 → 302 → 310 → 311 → 312 → 314 → 316 → 350, then the round-6
# synthesis (670) and the pipeline tail. Stops at the first failure; status lines in logs/ma2025_chain.status.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
source ${COCH_ENVS:-$COCH_ROOT/envs}/sc/bin/activate
export THREADS=16; source scripts/setup/threads.sh
ST=logs/ma2025_chain.status; : > $ST
step() {  # name, command...
  local name=$1; shift
  "$@" > logs/ma_${name}.log 2>&1; local rc=$?
  echo "$name EXIT $rc" >> $ST
  [ $rc -eq 0 ] || { echo "STOPPED at $name" >> $ST; exit $rc; }
}
step prep python -I scripts/setup/prep_ma2025.py
step 300 python scripts/300_external_sc_ingest.py ma2025
step 301 python scripts/301_external_sc_qc.py ma2025
step 302 python scripts/302_reference_mapping.py ma2025
step 310 python scripts/310_external_sc_response.py ma2025
step 311 bash scripts/setup/rjob.sh scripts/311_external_pseudobulk_de.R ma2025
step 312 python scripts/312_ma2025_rhythm_links.py ma2025
step 314 python scripts/314_ma2025_bulk.py ma2025
step 316 python scripts/316_ma2025_myeloid_b.py ma2025
step 350 python scripts/350_cochlea_immune_oscillations.py
step 670 python scripts/670_myeloid_synthesis.py
for s in 241_validation_matrix 360_conclusions 460_resource_tables 200_time_axis_audit 240_audit 97_plan_status 98_key_findings 99_results_manifest; do
  step $s python scripts/$s.py
done
echo DONE >> $ST
