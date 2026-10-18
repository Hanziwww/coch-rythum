#!/usr/bin/env bash
# Orchestrates Module 5.4 once the Ensembl GTF (needed for GSE88954 Ensembl IDs) has finished downloading.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
GTF=external/resources/Mus_musculus.GRCm39.113.gtf.gz
until [ -s "$GTF" ]; do sleep 60; done
P="nice -n 10 ${COCH_PYTHON:-python}"
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_external.log; }
log "GTF present"
$P scripts/70_external_prepare.py > logs/70_external_prepare.log 2>&1; log "70 exit $?"
nice -n 10 scripts/setup/rjob.sh scripts/71_external_rhythm.R > logs/71_external_rhythm.log 2>&1; log "71 exit $?"
for s in 72_phasor_decomposition 73_meta_cosinor 74_timetelling 75_deconvolution; do
  $P scripts/$s.py > logs/$s.log 2>&1; log "$s exit $?"
done
# BayesPrism for both tissues (waits for a running BM-only BayesPrism job to finish first)
while screen -ls | grep -q m5bp; do sleep 60; done
nice -n 10 scripts/setup/rjob.sh scripts/75b_bayesprism.R > logs/75b_bayesprism.log 2>&1; log "75b exit $?"
log "EXTERNAL_DONE"
