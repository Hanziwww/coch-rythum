# Public data

Public data are referenced here, not redistributed. `accessions.csv` lists

- the 24 data sets analysed in the article (Supplementary Table 20: accession, species, tissue, assay, design, use,
  figures, reference),
- 8 further data sets that the analysis code downloads
  for exploratory analyses whose tables are in `results/` but not in the article, and
- 12 reference resources (gene sets, regulons, orthology, motif rankings,
  ligand-target priors, genome annotation and sequences),

with the download route of each (`download`, `url`).

## Download

```bash
bash data/external/fetch_public_data.sh            # everything (about 9 GB)
bash data/external/fetch_public_data.sh geo        # one section; sections: genesets regulons annotation cistarget nichenet gwas geo r2 r2pilot r3 r6 r5 r4 promoters
```

Files go to `external/` in the project root (`code/scripts/config.yaml`: `paths.external`), in the layout the
scripts read: `external/geo/<accession>/` and `external/resources/`. Downloads resume, and finished files are skipped.

## Independent noise cohort (GSA)

The noise cohort (CRA019820, single-cell; CRA019827, bulk; Genome Sequence Archive, China National Center for
Bioinformation) is available as raw reads. The analyses used STARsolo 2.7.11b matrices (10x Genomics mm10-3.0.0
reference; filtered EmptyDrops-based cell calls) and a STAR gene-count table of the bulk libraries, placed in
`external/ma2025/`; `code/scripts/setup/prep_ma2025.py`, `300_external_sc_ingest.py` and `314_ma2025_bulk.py` describe the
expected layout.
