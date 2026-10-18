# Convert the PROGENy mouse model (.rda from the saezlab/progeny GitHub repository) to CSV for 41_pathway_activity.py.
f <- "external/resources/progeny_model_mouse_full.rda"
e <- new.env()
load(f, envir = e)
obj <- get(ls(e)[1], envir = e)
stopifnot(all(c("gene", "pathway", "weight", "p.value") %in% names(obj)))
write.csv(obj[, c("gene", "pathway", "weight", "p.value")], "external/resources/progeny_mouse.csv", row.names = FALSE)
cat("progeny mouse:", nrow(obj), "rows,", length(unique(obj$pathway)), "pathways\n")
