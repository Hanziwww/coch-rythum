#!/usr/bin/env bash
# Remaining dependent steps of Modules 3–6 (replaces chain_misc.sh; thread-capped): CCC rhythm after LIANA (m456);
# systemic cues after pathway activity (41 rerun with MGI orthologues, screen m3a); promoter-motif driver analysis;
# download-dependent analyses (MultiNicheNet, GWAS) as soon as their resources arrive (wait up to 6 h).
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
THREADS=4 source scripts/setup/threads.sh
P="nice -n 10 ${COCH_PYTHON:-python} -u"
R="nice -n 10 scripts/setup/rjob.sh"
RES=external/resources
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_misc.log; }
log "chain_misc2 started"
while screen -ls | grep -q "\.m456\s"; do sleep 60; done
$P scripts/64_ccc_rhythm.py > logs/64_ccc_rhythm.log 2>&1; log "64 exit $?"
while screen -ls | grep -q "\.m3a\s"; do sleep 60; done
$P scripts/82_systemic_cues_readiness.py > logs/82_systemic.log 2>&1; log "82 exit $?"
d44=0; d67=0; d84=0
for i in $(seq 1 360); do
  if [ $d44 = 0 ] && [ -s $RES/upstream1000.fa.gz ]; then
    $P scripts/44_promoter_motifs.py > logs/44_promoter_motifs.log 2>&1; log "44 (promoter motifs) exit $?"; d44=1
  fi
  if [ $d67 = 0 ] && [ -s $RES/ligand_target_matrix_nsga2r_final_mouse.rds ]; then
    $P scripts/67_export_sce.py > logs/67_export_sce.log 2>&1 && $R scripts/67_multinichenet.R > logs/67_multinichenet.log 2>&1
    log "67 exit $?"; d67=1
  fi
  if [ $d84 = 0 ] && [ -s $RES/gwas_catalog_associations.tsv ]; then
    $P scripts/84_gwas_overlap.py > logs/84_gwas_overlap.log 2>&1; log "84 exit $?"; d84=1
  fi
  [ "$d44$d67$d84" = 111 ] && break
  sleep 60
done
[ $d44 = 0 ] && log "44 skipped: promoter sequences unavailable"
[ $d67 = 0 ] && log "67 skipped: NicheNet ligand-target matrix unavailable"
[ $d84 = 0 ] && log "84 skipped: GWAS Catalog unavailable"
log "MISC_DONE"
