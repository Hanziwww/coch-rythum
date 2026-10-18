#!/usr/bin/env bash
# Final Module 1 → 4/5 steps: waits for the joint family of chain_dr4 (bm and cochlea families are complete), stops
# chain_dr4 before it recomputes the cochlea metrics, then 23 finalize → 53 Milo → 66 (BM-likeness) → 97/98/99.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
THREADS=6 source scripts/setup/threads.sh
P="nice -n 5 ${COCH_PYTHON:-python} -u"
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_dr2.log; }
until grep -q "joint stage C done" logs/chain_dr2.log; do sleep 30; done
for p in $(pgrep -f "chain_dr4.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "22_dr_metrics.py cochlea|21_dr_train.py cochlea"); do kill $p; done
log "chain_final: chain_dr4 stopped after the joint family"
$P scripts/23_dr_finalize.py > logs/23_dr_finalize.log 2>&1; log "23 exit $?"
nice -n 10 scripts/setup/rjob.sh scripts/53_milo_rhythm.R > logs/53_milo_rhythm.log 2>&1; log "53 exit $?"
$P scripts/66_state_continua.py > logs/66_state_continua.log 2>&1; log "66 exit $?"
$P scripts/97_plan_status.py > logs/97_plan_status.log 2>&1; log "97 exit $?"
$P scripts/98_key_findings.py > logs/98_key_findings.log 2>&1; log "98 exit $?"
$P scripts/99_results_manifest.py > logs/99_results_manifest.log 2>&1; log "99 exit $?"
log "DR_DONE"
