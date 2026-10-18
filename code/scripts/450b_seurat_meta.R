# Round 4, WP-E — export the author cell metadata of GSE234926 (P90 cochlear-duct Fbxo2+ nuclei; Seurat object) for
# label concordance in 450_atlas_concordance.py. Output: cache/r4_concordance/GSE234926_author_meta.csv
suppressPackageStartupMessages(library(SeuratObject))
root <- Sys.getenv("COCH_ROOT", ".")
f <- file.path(root, "external/geo/GSE234926/GSE234926_P90_Seurat_object.rds.gz")
out <- file.path(root, "cache/r4_concordance")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
tmp <- file.path(out, "GSE234926_P90.rds")
if (!file.exists(tmp)) system(paste("gzip -dc", shQuote(f), ">", shQuote(tmp)))
obj <- readRDS(tmp)
md <- obj@meta.data
md$cell <- rownames(md)
cat("cells", nrow(md), "\ncolumns", paste(colnames(md), collapse = ", "), "\n")
for (cn in colnames(md)) if (is.factor(md[[cn]]) || is.character(md[[cn]])) {
  u <- unique(md[[cn]]); if (length(u) <= 60) cat(cn, ":", paste(head(sort(as.character(u)), 60), collapse = " | "), "\n")
}
write.csv(md, file.path(out, "GSE234926_author_meta.csv"), row.names = FALSE)
unlink(tmp)
