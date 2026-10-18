#!/usr/bin/env bash
# Remaining dependent steps (Modules 3–6): CCC rhythm after LIANA; systemic cues after pathways; sccomp; optional
# download-dependent analyses (motifs, MultiNicheNet, GWAS) if their resources arrive; final MGI-ortholog rerun of 41.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
P="nice -n 10 ${COCH_PYTHON:-python} -u"
R="nice -n 10 scripts/setup/rjob.sh"
RES=external/resources
log() { echo "[$(date +%H:%M:%S)] $*" >> logs/chain_misc.log; }
$R scripts/51_abundance_models.R > logs/51_abundance_models.log 2>&1; log "51 (with sccomp) exit $?"
while screen -ls | grep -q "\.m456\s"; do sleep 60; done                      # LIANA (63) is the last step of m456
$P scripts/64_ccc_rhythm.py > logs/64_ccc_rhythm.log 2>&1; log "64 exit $?"
while screen -ls | grep -q "\.m3a\s"; do sleep 60; done                       # 41 pathway activity
$P scripts/82_systemic_cues_readiness.py > logs/82_systemic.log 2>&1; log "82 exit $?"
# optional analyses whose large downloads may still be running (wait up to 3 h)
for i in $(seq 1 180); do
  [ -s $RES/mm10_500bp_up_100bp_down_v10_clust.rankings.feather ] && [ -s $RES/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl ] && \
  [ -s $RES/ligand_target_matrix_nsga2r_final_mouse.rds ] && [ -s $RES/gwas_catalog_associations.tsv ] && break
  sleep 60
done
if [ -s $RES/mm10_500bp_up_100bp_down_v10_clust.rankings.feather ] && [ -s $RES/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl ]; then
  $R scripts/44_motif_enrichment.R > logs/44_motif_enrichment.log 2>&1; log "44 exit $?"
else log "44 skipped: cisTarget resources unavailable"; fi
if [ -s $RES/ligand_target_matrix_nsga2r_final_mouse.rds ]; then
  $P scripts/67_export_sce.py > logs/67_export_sce.log 2>&1 && $R scripts/67_multinichenet.R > logs/67_multinichenet.log 2>&1
  log "67 exit $?"
else log "67 skipped: NicheNet ligand-target matrix unavailable"; fi
if [ -s $RES/gwas_catalog_associations.tsv ]; then
  $P scripts/84_gwas_overlap.py > logs/84_gwas_overlap.log 2>&1; log "84 exit $?"
else log "84 skipped: GWAS Catalog unavailable"; fi
# rerun pathway activity with MGI 1:1 orthologues for CollecTRI (first pass used case conversion), then cues
$P scripts/41_pathway_activity.py > logs/41_pathway_activity_mgi.log 2>&1; log "41 (MGI) exit $?"
$P scripts/82_systemic_cues_readiness.py > logs/82_systemic.log 2>&1; log "82 rerun exit $?"
log "MISC_DONE"
