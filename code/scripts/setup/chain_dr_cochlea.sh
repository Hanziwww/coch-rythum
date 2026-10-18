#!/usr/bin/env bash
# Module 1, cochlea family only, run in parallel with chain_dr4 (bm → joint → cochlea, which then reuses the
# cached cochlea embeddings). For the family:
# metrics → top-2 → Stage C n_latent grid → metrics → seeds at best n_latent → metrics.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
THREADS=6 source scripts/setup/threads.sh
P="nice -n 5 ${COCH_PYTHON:-python} -u"
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_dr2.log; }
log "chain_dr_cochlea started"
for fam in cochlea; do
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
log "cochlea family (parallel chain) done"
