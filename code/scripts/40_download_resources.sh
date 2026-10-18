#!/usr/bin/env bash
# Download prior-knowledge resources (Module 3), external circadian datasets (Module 5.4) and round-2 validation data.
# Routing: Broad/Ensembl/NCBI/OmniPath/MGI/Aerts-lab direct; GitHub raw through the academic proxy only.
set -uo pipefail
ROOT=${COCH_ROOT:-.}
RES=$ROOT/external/resources
EXT=$ROOT/external/geo
mkdir -p "$RES" "$EXT"
unset http_proxy https_proxy

get() {  # url dest [proxy] — resumable, retried up to 30 times
  local url=$1 dest=$2 i
  if [ -s "$dest" ]; then echo "exists $(basename "$dest")"; return 0; fi
  for i in $(seq 1 30); do
    if [ "${3:-}" = proxy ]; then
      ( curl -fsSL --http1.1 -C - -m 1800 -o "$dest.part" "$url" ) && mv "$dest.part" "$dest" && break
    else
      curl -fsSL --http1.1 -C - -m 1800 -o "$dest.part" "$url" && mv "$dest.part" "$dest" && break
    fi
    sleep 5
  done
  if [ -s "$dest" ]; then echo "ok     $(basename "$dest") $(du -h "$dest" | cut -f1)"; else echo "FAILED $url"; fi
}

SECTION=${1:-all}
run() { [ "$SECTION" = all ] || [ "$SECTION" = "$1" ]; }

# --- gene sets (MSigDB mouse collections, release 2024.1.Mm)
if run genesets; then
MS=https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Mm
get $MS/mh.all.v2024.1.Mm.symbols.gmt          $RES/msigdb_mh_hallmark.gmt
get $MS/m2.cp.reactome.v2024.1.Mm.symbols.gmt  $RES/msigdb_m2_reactome.gmt
get $MS/m5.go.bp.v2024.1.Mm.symbols.gmt        $RES/msigdb_m5_gobp.gmt
fi
# --- TF regulons (CollecTRI, human symbols; mapped to mouse with MGI orthologs) and PROGENy mouse model
if run regulons; then
get "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&license=academic" $RES/collectri_human.tsv
get https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt $RES/mgi_hom_mouse_human.rpt
fi
if run annotation; then
get https://raw.githubusercontent.com/saezlab/progeny/master/data/model_mouse_full.rda $RES/progeny_model_mouse_full.rda proxy
# --- gene annotation (chromosomes, biotypes)
get https://ftp.ensembl.org/pub/release-113/gtf/mus_musculus/Mus_musculus.GRCm39.113.gtf.gz $RES/Mus_musculus.GRCm39.113.gtf.gz
fi
if run cistarget; then
# --- cisTarget (mm10, 500 bp up / 100 bp down promoter rankings) + motif annotations
CT=https://resources.aertslab.org/cistarget
get $CT/databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/gene_based/mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather $RES/mm10_500bp_up_100bp_down_v10_clust.rankings.feather
get $CT/motif2tf/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl $RES/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl
fi

# --- NicheNet v2 mouse prior (Zenodo) and GWAS Catalog associations (EBI FTP)
if run nichenet; then
get https://zenodo.org/record/7074291/files/lr_network_mouse_21122021.rds $RES/lr_network_mouse_21122021.rds
get https://zenodo.org/record/7074291/files/ligand_target_matrix_nsga2r_final_mouse.rds $RES/ligand_target_matrix_nsga2r_final_mouse.rds
fi
if run gwas; then
get https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-associations-full.zip $RES/gwas_catalog_associations_full.zip
[ -s $RES/gwas_catalog_associations.tsv ] || (cd $RES && unzip -p gwas_catalog_associations_full.zip '*.tsv' > gwas_catalog_associations.tsv && echo unzipped)
fi

# --- external circadian datasets (GEO supplementary files + series matrices)
geo() {  # GSE
  local gse=$1 stub=${1:0:${#1}-3}nnn
  mkdir -p "$EXT/$gse"
  local list
  list=$(curl -fsSL -m 120 "https://ftp.ncbi.nlm.nih.gov/geo/series/$stub/$gse/suppl/" | grep -oE 'href="[^"]+"' | cut -d'"' -f2 | grep -v '^/' | grep -v '^?' )
  for f in $list; do get "https://ftp.ncbi.nlm.nih.gov/geo/series/$stub/$gse/suppl/$f" "$EXT/$gse/$f"; done
  get "https://ftp.ncbi.nlm.nih.gov/geo/series/$stub/$gse/matrix/${gse}_series_matrix.txt.gz" "$EXT/$gse/${gse}_series_matrix.txt.gz"
}
if run geo; then for g in GSE88954 GSE198770 GSE86619 GSE54652 GSE54650; do echo "== $g"; geo $g; done; fi
# --- round 2 external validation (explicit files; NCBI through the proxy)
if run r2; then
G=https://ftp.ncbi.nlm.nih.gov/geo
mkdir -p "$EXT/GSE198770" "$EXT/GSE54650" "$EXT/GSE25585" "$EXT/GSE192616"
for t in gut_ab liver_ab lung_ab skin_dt; do
  get $G/series/GSE198nnn/GSE198770/suppl/GSE198770_${t}_normalized_counts.csv.gz "$EXT/GSE198770/GSE198770_${t}_normalized_counts.csv.gz" proxy
done
get $G/series/GSE54nnn/GSE54650/matrix/GSE54650_series_matrix.txt.gz "$EXT/GSE54650/GSE54650_series_matrix.txt.gz" proxy
get $G/series/GSE25nnn/GSE25585/matrix/GSE25585_series_matrix.txt.gz "$EXT/GSE25585/GSE25585_series_matrix.txt.gz" proxy
get $G/platforms/GPL6nnn/GPL6246/annot/GPL6246.annot.gz "$RES/GPL6246.annot.gz" proxy
get $G/series/GSE192nnn/GSE192616/suppl/GSE192616_sc_adata_oct22m.h5ad.gz "$EXT/GSE192616/GSE192616_sc_adata_oct22m.h5ad.gz" proxy
fi
# --- round 2 treatment-response / spatial pilots (WP6; small processed files only; series matrices for metadata)
if run r2pilot; then
G=https://ftp.ncbi.nlm.nih.gov/geo
for g in GSE107086 GSE312646 GSE312476; do mkdir -p "$EXT/$g"; done
get $G/series/GSE107nnn/GSE107086/suppl/GSE107086_raw_counts.csv.gz "$EXT/GSE107086/GSE107086_raw_counts.csv.gz"
get $G/series/GSE107nnn/GSE107086/matrix/GSE107086_series_matrix.txt.gz "$EXT/GSE107086/GSE107086_series_matrix.txt.gz"
get $G/series/GSE312nnn/GSE312646/suppl/GSE312646_GeoMX_table.csv.gz "$EXT/GSE312646/GSE312646_GeoMX_table.csv.gz"
get $G/series/GSE312nnn/GSE312646/matrix/GSE312646_series_matrix.txt.gz "$EXT/GSE312646/GSE312646_series_matrix.txt.gz"
get $G/series/GSE312nnn/GSE312476/suppl/GSE312476_RAW.tar "$EXT/GSE312476/GSE312476_RAW.tar"
get $G/series/GSE312nnn/GSE312476/matrix/GSE312476_series_matrix.txt.gz "$EXT/GSE312476/GSE312476_series_matrix.txt.gz"
fi
# --- round 3 public response data (WP2): LPS × time of day in BM/blood neutrophils; cochlear immune cells after noise
#     (NCBI through the proxy: 8 MB/s vs 17 KB/s direct on 2026-10-12)
if run r3; then
G=https://ftp.ncbi.nlm.nih.gov/geo
for g in GSE312298 GSE167078 GSE346691; do mkdir -p "$EXT/$g"; done
get $G/series/GSE312nnn/GSE312298/suppl/GSE312298_DESeq2_All_togother_umi_counts_20220307_105325.xlsx "$EXT/GSE312298/GSE312298_DESeq2_All_togother_umi_counts_20220307_105325.xlsx" proxy
get $G/series/GSE312nnn/GSE312298/matrix/GSE312298_series_matrix.txt.gz "$EXT/GSE312298/GSE312298_series_matrix.txt.gz" proxy
get $G/series/GSE167nnn/GSE167078/suppl/GSE167078_RAW.tar "$EXT/GSE167078/GSE167078_RAW.tar" proxy
get $G/series/GSE167nnn/GSE167078/matrix/GSE167078_series_matrix.txt.gz "$EXT/GSE167078/GSE167078_series_matrix.txt.gz" proxy
get $G/series/GSE346nnn/GSE346691/suppl/GSE346691_RAW.tar "$EXT/GSE346691/GSE346691_RAW.tar" proxy
get $G/series/GSE346nnn/GSE346691/matrix/GSE346691_series_matrix.txt.gz "$EXT/GSE346691/GSE346691_series_matrix.txt.gz" proxy
fi
if run r6; then
# round 6 (myeloid): neutrophil state references — Xie 2020 (GSE137540), NeuMAP (GSE266518; subset later via the samples
# map), Ballesteros 2020 (GSE142754; blood neutrophils ZT5 vs ZT13 + tissues), neutrophils across tissues / ageing (GSE200145)
G=https://ftp.ncbi.nlm.nih.gov/geo
for g in GSE142754 GSE200145; do
  mkdir -p "$EXT/$g"
  get $G/series/${g:0:${#g}-3}nnn/$g/matrix/${g}_series_matrix.txt.gz "$EXT/$g/${g}_series_matrix.txt.gz" proxy
done
mkdir -p "$EXT/GSE137540" "$EXT/GSE266518"          # multi-platform series: one matrix per GPL
for f in GSE137540-GPL17021 GSE137540-GPL24247 GSE137540-GPL24676; do
  get $G/series/GSE137nnn/GSE137540/matrix/${f}_series_matrix.txt.gz "$EXT/GSE137540/${f}_series_matrix.txt.gz" proxy
done
for f in GSE266518-GPL21103 GSE266518-GPL24247 GSE266518-GPL30172; do
  get $G/series/GSE266nnn/GSE266518/matrix/${f}_series_matrix.txt.gz "$EXT/GSE266518/${f}_series_matrix.txt.gz" proxy
done
get $G/series/GSE266nnn/GSE266518/suppl/GSE266518_samples_map_updated_GEO.csv.gz "$EXT/GSE266518/GSE266518_samples_map_updated_GEO.csv.gz" proxy
get $G/series/GSE200nnn/GSE200145/suppl/GSE200145_count_matrix.csv.gz "$EXT/GSE200145/GSE200145_count_matrix.csv.gz" proxy
get $G/series/GSE137nnn/GSE137540/suppl/GSE137540_RAW.tar "$EXT/GSE137540/GSE137540_RAW.tar" proxy
get $G/series/GSE142nnn/GSE142754/suppl/GSE142754_RAW.tar "$EXT/GSE142754/GSE142754_RAW.tar" proxy
get $G/series/GSE266nnn/GSE266518/suppl/GSE266518_RAW.tar "$EXT/GSE266518/GSE266518_RAW.tar" proxy
fi
if run r5; then
# round 5 (B cells): ImmGen sorted populations (microarray GSE15907 + platform annotation; ULI RNA-seq GSE109125),
# IgA+ plasma cells by time of day (GSE175635) and with B-cell Bmal1 KO (GSE175609), Cry1/2 KO bone marrow (GSE87467)
G=https://ftp.ncbi.nlm.nih.gov/geo
for g in GSE15907 GSE109125 GSE175609 GSE175635 GSE87467; do
  mkdir -p "$EXT/$g"
  get $G/series/${g:0:${#g}-3}nnn/$g/matrix/${g}_series_matrix.txt.gz "$EXT/$g/${g}_series_matrix.txt.gz" proxy
done
get $G/platforms/GPL6nnn/GPL6246/annot/GPL6246.annot.gz "$EXT/GSE15907/GPL6246.annot.gz" proxy
get $G/series/GSE109nnn/GSE109125/suppl/GSE109125_Gene_count_table_GENCODE_vM25.csv.gz "$EXT/GSE109125/GSE109125_Gene_count_table_GENCODE_vM25.csv.gz" proxy
get $G/series/GSE109nnn/GSE109125/suppl/GSE109125_readme_GENCODE_vM25.txt "$EXT/GSE109125/GSE109125_readme_GENCODE_vM25.txt" proxy
get $G/series/GSE175nnn/GSE175609/suppl/GSE175609_Penny_and_Hepworth_readcount_genename.xls.gz "$EXT/GSE175609/GSE175609_readcount_genename.xls.gz" proxy
get $G/series/GSE175nnn/GSE175609/suppl/GSE175609_Penny_and_Hepworth_DESeq_Normalised_Counts_sheet.xlsx "$EXT/GSE175609/GSE175609_DESeq_normalised.xlsx" proxy
get $G/series/GSE175nnn/GSE175635/suppl/GSE175635_Peny_and_Hepworth_IgA_PC_Circadian_genes.readcount.xlsx "$EXT/GSE175635/GSE175635_IgA_PC_circadian_readcount.xlsx" proxy
get $G/series/GSE87nnn/GSE87467/suppl/GSE87467_RAW.tar "$EXT/GSE87467/GSE87467_RAW.tar" proxy
fi
if run r4; then
# round 4: cross-species (baboon diurnal atlas, human diurnal neutrophils), external cochlear atlases (ageing, adult
# cochlear duct / hair cells, adult mouse + human fetal cochlea), adult human utricle. All < 1 GB → kept on the analysis server.
G=https://ftp.ncbi.nlm.nih.gov/geo
for g in GSE98965 GSE136632 GSE274279 GSE234926 GSE283534 GSE135913 GSE244055; do
  mkdir -p "$EXT/$g"
  get $G/series/${g:0:${#g}-3}nnn/$g/matrix/${g}_series_matrix.txt.gz "$EXT/$g/${g}_series_matrix.txt.gz" proxy
done
get $G/series/GSE98nnn/GSE98965/suppl/GSE98965_baboon_tissue_expression_FPKM.csv.gz "$EXT/GSE98965/GSE98965_baboon_tissue_expression_FPKM.csv.gz" proxy
get $G/series/GSE136nnn/GSE136632/suppl/GSE136632_normalized_cpm_humanRNAseq.csv.gz "$EXT/GSE136632/GSE136632_normalized_cpm_humanRNAseq.csv.gz" proxy
for g in GSE274279 GSE234926 GSE283534 GSE244055; do
  get $G/series/${g:0:${#g}-3}nnn/$g/suppl/${g}_RAW.tar "$EXT/$g/${g}_RAW.tar" proxy
done
for s in GSM4037815_m_adult_epi GSM4037816_m_adult_neg GSM4037817_m_adult_epi2 GSM4037818_m_adult_neg2          GSM4037819_fhu_15_pos GSM4037820_fhu_17 GSM4037821_fhu_23_pos; do
  for f in barcodes.tsv.gz genes.tsv.gz matrix.mtx.gz; do
    get $G/samples/GSM4037nnn/${s%%_*}/suppl/${s}_$f "$EXT/GSE135913/${s}_$f" proxy
  done
done
for s in GSM4037815_m_adult_epi GSM4037816_m_adult_neg GSM4037817_m_adult_epi2 GSM4037818_m_adult_neg2; do
  get $G/samples/GSM4037nnn/${s%%_*}/suppl/${s}_CellFindR_alg_matrix.csv.gz "$EXT/GSE135913/${s}_CellFindR_alg_matrix.csv.gz" proxy
done
get $G/series/GSE234nnn/GSE234926/suppl/GSE234926_P90_Seurat_object.rds.gz "$EXT/GSE234926/GSE234926_P90_Seurat_object.rds.gz" proxy
fi
ls -la "$RES" "$EXT"/*
echo "DOWNLOADS_DONE"
