"""Round 3, WP1 — train the scArches-ready scANVI reference of the cochlear atlas (lib/refmap.py) once; it is reused to
map Ma 2025 (CRA019820), GSE167078 and GSE346691 cells onto the gold-standard labels.
Output: cache/r3_refmap/cochlea_scanvi/ (model), results/r3_refmap/reference_model.csv (settings, self-accuracy)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import refmap  # noqa: E402
from lib.io import log, write_csv  # noqa: E402

out = refmap.train_reference(force="--force" in sys.argv)
import scvi  # noqa: E402

ref = refmap.load_reference()
genes = [g for g in refmap.HVG_FILE.read_text().split() if g in ref.var_names]
lv = scvi.model.SCANVI.load(str(out), adata=ref[:, genes].copy())
pred = lv.predict()
acc = (pred == ref.obs["cell_type"].to_numpy())
per = pd.DataFrame({"cell_type": ref.obs["cell_type"].to_numpy(), "correct": acc}).groupby("cell_type")["correct"].agg(["size", "mean"]).reset_index()
per.columns = ["cell_type", "n_cells", "self_prediction_accuracy"]
per.loc[len(per)] = {"cell_type": "ALL", "n_cells": int(len(acc)), "self_prediction_accuracy": float(acc.mean())}
per["settings"] = ("scVI n_latent 30, 2 layers, 128 hidden, NB, batch = sample, layer norm, encode covariates, dropout 0.2; "
                   "SCANVI 20 epochs; genes = A__seurat_v3_5000_ex0")
write_csv(per, "r3_refmap", "reference_model.csv", "scANVI reference of the cochlear atlas: settings and self-prediction accuracy",
          required=["cell_type", "n_cells", "self_prediction_accuracy"])
log("done")
