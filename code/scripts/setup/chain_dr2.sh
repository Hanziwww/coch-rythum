#!/usr/bin/env bash
# Module 1 continuation: waits for Stage B training (chain_dr) and the running BM Stage C, then for every family:
# metrics → top-2 → Stage C n_latent grid → metrics → seeds at best n_latent → metrics; finally 23 finalize,
# 53 Milo and the BM-likeness rerun of 66.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
P="nice -n 5 ${COCH_PYTHON:-python} -u"
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_dr2.log; }
while screen -ls | grep -qE "chain_dr\b|\.chain_dr\s|gpu_bmC"; do sleep 60; done
log "stage B training finished"
for fam in bm joint cochlea; do
  HVG=$($P scripts/20b_select_features.py $fam 2>/dev/null | tail -1)
  $P scripts/22_dr_metrics.py $fam B__pca_nl30_s0 > logs/22_dr_metrics_B_$fam.log 2>&1; log "$fam metrics B exit $?"
  METHODS=$($P scripts/20c_select_stageC.py $fam top2 2>/dev/null | tail -1)
  log "$fam stage C methods: $METHODS (hvg $HVG)"
  $P scripts/21_dr_train.py $fam C $HVG $METHODS > logs/21_dr_train_${fam}_C.log 2>&1
  $P scripts/22_dr_metrics.py $fam B__pca_nl30_s0 > logs/22_dr_metrics_C_$fam.log 2>&1
  for m in $METHODS; do
    NL=$($P scripts/20c_select_stageC.py $fam bestnlat $m 2>/dev/null | tail -1)
    log "$fam $m best n_latent $NL"
    BEST_NLAT=$NL $P scripts/21_dr_train.py $fam C $HVG $m >> logs/21_dr_train_${fam}_C.log 2>&1
  done
  $P scripts/22_dr_metrics.py $fam B__pca_nl30_s0 > logs/22_dr_metrics_C2_$fam.log 2>&1; log "$fam stage C done"
done
$P scripts/23_dr_finalize.py > logs/23_dr_finalize.log 2>&1; log "23 exit $?"
nice -n 10 scripts/setup/rjob.sh scripts/53_milo_rhythm.R > logs/53_milo_rhythm.log 2>&1; log "53 exit $?"
$P scripts/66_state_continua.py > logs/66_state_continua.log 2>&1; log "66 exit $?"
log "DR_DONE"
