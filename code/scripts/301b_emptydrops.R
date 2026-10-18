#!/usr/bin/env Rscript
# Round 3, WP1 — cell calling for one raw (unfiltered) external 10x sample with DropletUtils::emptyDrops.
# Input directory (argument): matrix.mtx (genes × barcodes), barcodes.tsv, genes.tsv written by 301_external_sc_qc.py.
# Cells = FDR < 0.001 with lower = 100 UMIs (emptyDrops also retains every barcode above the knee via retain = NULL
# → knee point). Output: <dir>/cells.csv (barcode, total, FDR).
suppressPackageStartupMessages({ library(DropletUtils); library(Matrix); library(data.table) })
d <- commandArgs(trailingOnly = TRUE)[1]
m <- as(readMM(file.path(d, "matrix.mtx")), "CsparseMatrix")
colnames(m) <- fread(file.path(d, "barcodes.tsv"), header = FALSE)[[1]]
rownames(m) <- make.unique(fread(file.path(d, "genes.tsv"), header = FALSE)[[1]])
set.seed(100)
e <- emptyDrops(m, lower = 100, niters = 10000)
is_cell <- !is.na(e$FDR) & e$FDR < 0.001
fwrite(data.table(barcode = colnames(m)[is_cell], total = e$Total[is_cell], FDR = e$FDR[is_cell]), file.path(d, "cells.csv"))
message(sprintf("%s: %d cells of %d barcodes", basename(d), sum(is_cell), ncol(m)))
