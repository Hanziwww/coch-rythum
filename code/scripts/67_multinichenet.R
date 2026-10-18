#!/usr/bin/env Rscript
# Module 5.2 — MultiNicheNet: prioritised ligand → receptor → target links for rest (ZT3/7/11) vs active (ZT15/19/23)
# phase contrasts per tissue (3 vs 3 pooled samples; pseudobulk edgeR DE via muscat; NicheNet v2 mouse prior).
# Senders/receivers: gold-standard cell types with >= 10 cells in every sample of the tissue (env IMMUNE_RECEIVERS=1
# restricts receivers to immune/haematopoietic types and MAX_CELLS caps cells per sample × cell type — used for the
# cochlea, whose full run exceeded the container's ~90 GB memory limit).
# Outputs (results/67_multinichenet/): multinichenet_prioritization_<tissue>.csv, multinichenet_ligand_activity_<tissue>.csv
suppressPackageStartupMessages({ library(multinichenetr); library(SingleCellExperiment); library(dplyr)
  library(data.table); library(Matrix) })
source("scripts/lib/cosinor.R")
out_dir <- file.path(RESULTS, "67_multinichenet"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
res_dir <- file.path(ROOT, "external/resources")
lr_network <- readRDS(file.path(res_dir, "lr_network_mouse_21122021.rds")) %>%
  dplyr::rename(ligand = from, receptor = to) %>% distinct(ligand, receptor)
ligand_target_matrix <- readRDS(file.path(res_dir, "ligand_target_matrix_nsga2r_final_mouse.rds"))
rds <- file.path(CACHE, "67_multinichenet/counts_dgC.rds")
if (file.exists(rds)) mm <- readRDS(rds) else {
  mm <- as(Matrix::readMM(file.path(CACHE, "67_multinichenet/counts.mtx")), "CsparseMatrix")   # 67_export_sce.py
  saveRDS(mm, rds)
}
meta <- fread(file.path(CACHE, "67_multinichenet/cells.csv"))
# syntactically valid, ASCII cell-type ids (MultiNicheNet requirement); original labels restored in the outputs
safe_id <- function(x) gsub("^_|_$", "", gsub("[^A-Za-z0-9]+", "_", iconv(x, "UTF-8", "ASCII//TRANSLIT")))
id_map <- unique(data.table(cell_type = meta$cell_type, celltype_id = safe_id(meta$cell_type)))
stopifnot(!anyDuplicated(id_map$celltype_id))
restore <- function(dt) {
  for (col in intersect(c("sender", "receiver", "cluster_id"), names(dt)))
    dt[[col]] <- id_map$cell_type[match(dt[[col]], id_map$celltype_id)]
  dt
}
genes <- fread(file.path(CACHE, "67_multinichenet/genes.csv"))$gene
TISSUES <- strsplit(Sys.getenv("TISSUES", "bm,cochlea"), ",")[[1]]
MAX_CELLS <- as.integer(Sys.getenv("MAX_CELLS", "0"))           # cap per sample × cell type (memory; 0 = all)
IMMUNE_RECEIVERS <- Sys.getenv("IMMUNE_RECEIVERS", "0") == "1"  # plan: immune receivers, all senders
NC <- as.integer(Sys.getenv("NCORES", "8"))
immune_rx <- "B_Cell|Pre_B|Pro_B|T_Cell|NK_Cell|Neutrophil|Monocyte|Macrophage|Mast|pDC|Granulocyte|HSC"
for (tis in TISSUES) {
  idx <- which(meta$tissue == tis)
  if (MAX_CELLS > 0) {
    set.seed(1)
    dt <- data.table(i = idx, s = meta$sample_id[idx], c = meta$cell_type[idx])
    idx <- sort(dt[, .(i = if (.N > MAX_CELLS) sample(i, MAX_CELLS) else i), by = .(s, c)]$i)
    message(tis, ": capped at ", MAX_CELLS, " cells per sample × cell type -> ", length(idx), " cells")
  }
  cm <- as(Matrix::t(mm[idx, ]), "CsparseMatrix")
  dimnames(cm) <- list(genes, meta$cell_id[idx])
  sce <- SingleCellExperiment(assays = list(counts = cm), colData = as.data.frame(meta[idx]))
  sce <- scuttle::logNormCounts(sce)
  sce$group_id <- ifelse(sce$zt %in% c(3, 7, 11), "rest", "active")
  sce$celltype_id <- safe_id(sce$cell_type)
  message(tis, ": ", ncol(sce), " cells; counts class ", class(counts(sce))[1], "; total UMI ", sum(counts(sce)))
  ok <- as.data.table(colData(sce))[, .N, by = .(sample_id, celltype_id)][, .(min = min(N), ns = .N), by = celltype_id][ns == 6 & min >= 10, celltype_id]
  sce <- sce[, sce$celltype_id %in% ok]
  contrasts_oi <- c("'rest-active','active-rest'")
  contrast_tbl <- tibble(contrast = c("rest-active", "active-rest"), group = c("rest", "active"))
  senders_oi <- receivers_oi <- sort(unique(sce$celltype_id))
  if (IMMUNE_RECEIVERS) receivers_oi <- grep(immune_rx, senders_oi, value = TRUE)
  min_cells <- 10
  abundance_info <- get_abundance_info(sce, "sample_id", "group_id", "celltype_id", min_cells, senders_oi, receivers_oi,
                                       batches = NA)
  frq_list <- get_frac_exprs(sce, "sample_id", "celltype_id", "group_id", batches = NA, min_cells = min_cells,
                             fraction_cutoff = 0.05, min_sample_prop = 0.5)
  genes_oi <- frq_list$expressed_df %>% filter(expressed == TRUE) %>% pull(gene) %>% unique()
  message(tis, ": expressed genes ", length(genes_oi))
  stopifnot(length(genes_oi) > 100)
  sce <- sce[genes_oi, ]
  aei <- process_abundance_expression_info(sce, "sample_id", "group_id", "celltype_id", min_cells, senders_oi,
                                           receivers_oi, lr_network, batches = NA, frq_list = frq_list,
                                           abundance_info = abundance_info)
  DE_info <- get_DE_info(sce, "sample_id", "group_id", "celltype_id", batches = NA, covariates = NA,
                         contrasts_oi = contrasts_oi, min_cells = min_cells, expressed_df = frq_list$expressed_df)
  celltype_de <- DE_info$celltype_de$de_output_tidy
  srd <- combine_sender_receiver_de(sender_de = celltype_de, receiver_de = celltype_de, senders_oi = senders_oi,
                                    receivers_oi = receivers_oi, lr_network = lr_network)
  lat <- get_ligand_activities_targets_DEgenes(receiver_de = celltype_de, receivers_oi = intersect(receivers_oi, celltype_de$cluster_id),
                                               ligand_target_matrix = ligand_target_matrix, logFC_threshold = 0.5,
                                               p_val_threshold = 0.05, p_val_adj = FALSE, top_n_target = 250,
                                               verbose = FALSE, n.cores = NC)
  grouping_tbl <- as_tibble(colData(sce))[, c("sample_id", "group_id")] %>% distinct() %>% dplyr::rename(sample = sample_id, group = group_id)
  pt <- generate_prioritization_tables(sender_receiver_info = aei$sender_receiver_info, sender_receiver_de = srd,
                                       ligand_activities_targets_DEgenes = lat, contrast_tbl = contrast_tbl,
                                       sender_receiver_tbl = srd %>% distinct(sender, receiver), grouping_tbl = grouping_tbl,
                                       scenario = "regular", fraction_cutoff = 0.05,
                                       abundance_data_receiver = aei$abundance_data_receiver,
                                       abundance_data_sender = aei$abundance_data_sender, ligand_activity_down = FALSE)
  fwrite(restore(as.data.table(pt$group_prioritization_tbl))[, tissue := tis],
         file.path(out_dir, sprintf("multinichenet_prioritization_%s.csv", tis)))
  fwrite(restore(as.data.table(lat$ligand_activities))[, tissue := tis],
         file.path(out_dir, sprintf("multinichenet_ligand_activity_%s.csv", tis)))
  message(tis, ": prioritised interactions ", nrow(pt$group_prioritization_tbl))
}
message("done")
