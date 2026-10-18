#!/usr/bin/env bash
# Download the public data analysed in this study (data/external/accessions.csv) into $COCH_ROOT/external, where the
# analysis scripts read them (code/scripts/config.yaml: paths.external). Public data are referenced, not redistributed.
# Usage, from the release folder:   bash data/external/fetch_public_data.sh [section]
#   sections (default: all): genesets regulons annotation cistarget nichenet gwas geo r2 r2pilot r3 r6 r5 r4 promoters
# About 9 GB; reruns resume and skip finished files.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export COCH_ROOT=${COCH_ROOT:-$(cd "$HERE/../.." && pwd)}
SECTION=${1:-all}
if [ "$SECTION" != promoters ]; then
  bash "$COCH_ROOT/code/scripts/40_download_resources.sh" "$SECTION"
fi
if [ "$SECTION" = all ] || [ "$SECTION" = promoters ]; then        # promoter motifs (44_promoter_motifs.py)
  RES=$COCH_ROOT/external/resources
  mkdir -p "$RES"
  for f in database/refGene.txt.gz bigZips/upstream1000.fa.gz; do
    out="$RES/$(basename "$f")"
    [ -s "$out" ] || curl -fsSL --retry 5 -C - -o "$out" "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/$f"
  done
fi
cat <<'EOF'
Not downloaded: the independent noise cohort (GSA CRA019820, single-cell; CRA019827, bulk) is public as raw reads at
https://ngdc.cncb.ac.cn/gsa/ . Quantify them with STARsolo 2.7.11b (10x Genomics mm10-3.0.0 reference) and STAR
GeneCounts and place the matrices in external/ma2025/ (see code/scripts/setup/prep_ma2025.py).
EOF
