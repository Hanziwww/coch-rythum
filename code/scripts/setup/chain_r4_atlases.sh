#!/usr/bin/env bash
# Round 4, WP-E: ingest → QC → reference mapping for the external adult cochlear atlases (sequential; GPU for 302).
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
source scripts/setup/threads.sh 8
PY=${COCH_PYTHON:-python}
bash scripts/setup/rjob.sh scripts/450b_seurat_meta.R > logs/450b_seurat_meta.log 2>&1; echo "[$(date +%T)] 450b exit $?" >> logs/chain_r4_atlases.log
for k in gse283534 gse135913 gse234926 gse274279; do
  for s in 300_external_sc_ingest 301_external_sc_qc 302_reference_mapping; do
    $PY scripts/$s.py $k > logs/${s}_$k.log 2>&1
    rc=$?
    echo "[$(date +%T)] $s $k exit $rc" >> logs/chain_r4_atlases.log
    [ $rc -ne 0 ] && break
  done
done
echo "[$(date +%T)] CHAIN_R4_ATLASES_DONE" >> logs/chain_r4_atlases.log
