#!/usr/bin/env bash
# Pipeline tail after any new analysis: 900 index -> 241 -> 360 -> 200 -> 240 -> 97 -> 98 -> 99.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
THREADS=6 source scripts/setup/threads.sh
P="nice -n 5 ${COCH_PYTHON:-python} -u"
L=logs/chain_tail.log
: > "$L"
for s in "900_figure_data.py --index-only" 241_validation_matrix.py 360_conclusions.py 200_time_axis_audit.py \
         240_audit.py 97_plan_status.py 98_key_findings.py 99_results_manifest.py; do
  echo "== $s" >> "$L"
  $P scripts/$s >> "$L" 2>&1 || { echo "FAILED $s" >> "$L"; exit 1; }
done
echo "TAIL_DONE" >> "$L"
