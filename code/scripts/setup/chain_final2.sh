#!/usr/bin/env bash
# Final steps after the Milo fix: 53 Milo → 66 (BM-likeness) → 97/98/99.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
THREADS=6 source scripts/setup/threads.sh
P="nice -n 5 ${COCH_PYTHON:-python} -u"
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_dr2.log; }
log "chain_final2 started (Milo with sparse neighbourhood counts)"
nice -n 10 scripts/setup/rjob.sh scripts/53_milo_rhythm.R > logs/53_milo_rhythm.log 2>&1; log "53 exit $?"
$P scripts/66_state_continua.py > logs/66_state_continua.log 2>&1; log "66 exit $?"
$P scripts/97_plan_status.py > logs/97_plan_status.log 2>&1; log "97 exit $?"
$P scripts/98_key_findings.py > logs/98_key_findings.log 2>&1; log "98 exit $?"
$P scripts/99_results_manifest.py > logs/99_results_manifest.log 2>&1; log "99 exit $?"
log "DR_DONE"
