#!/usr/bin/env bash
# Round 2 — final chain after 211 → 222 → 231. Sections (argument; default all):
#   tierc  Tier C downstream (exploratory): enrichment / PSEA, motifs, catalogue (needs 211 only)
#   final  pilots re-run, validation matrix, feasibility, audits, plan status, key findings, manifest (needs 222, 231)
# Run from the project root inside screen.
set -uo pipefail
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
PY=${COCH_PYTHON:-python}
THREADS=4 source scripts/setup/threads.sh
SECTION=${1:-all}
run_py() { echo "[$(date +%T)] $1"; nice -n 5 $PY -u "scripts/$1" > "logs/${1%.py}${2:-}.log" 2>&1 || echo "FAILED $1"; }
if [ "$SECTION" = all ] || [ "$SECTION" = tierc ]; then
  ( export TIERC=1 NJOBS=4; run_py 42_rhythm_enrichment.py _tierC ) &
  ( export TIERC=1 NCORES=4; nice -n 5 scripts/setup/rjob.sh scripts/44_motif_enrichment.R > logs/44_motif_enrichment_tierC.log 2>&1 || echo "FAILED 44 TIERC" ) &
  run_py 212_tierC_downstream.py
  wait
  echo "[$(date +%T)] R2_TIERC_DONE"
fi
if [ "$SECTION" = all ] || [ "$SECTION" = final ]; then
  run_py 252_pilot_noise_linking.py          # re-run with Tier C available
  run_py 241_validation_matrix.py
  run_py 250_feasibility.py
  run_py 200_time_axis_audit.py
  run_py 240_audit.py
  run_py 97_plan_status.py
  run_py 98_key_findings.py
  run_py 99_results_manifest.py
  echo "[$(date +%T)] R2_FINAL_DONE"
fi
