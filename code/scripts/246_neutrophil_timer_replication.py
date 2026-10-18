"""Round 2, WP5 — replication of the neutrophil-timer sign result (77): cochlear neutrophil rhythms predicted the
blood-neutrophil ZT13−ZT5 change (GSE86619) positively, temporal-BM neutrophils negatively.

Replication with external replicated time series: the ZT13−ZT5 difference predicted by each external unit's cosinor
fit, Δ̂ = a(cos 13ω − cos 5ω) + b(sin 13ω − sin 5ω) (internal clock-aligned ZT, offsets from 230), is correlated
(Spearman) with the observed blood-neutrophil WT ZT13−ZT5 log2 fold change, over all shared genes and over
neutrophil-specific genes (markers of our neutrophil states; 00_audit/label_markers.csv). Units: GSE198770 BM, gut,
liver, lung, skin (control and neutrophil-depleted), GSE88954 cochlea (sham, ADX). Expected if bulk BM behaves like our
temporal-BM neutrophils: negative correlation for BM controls; loss of the neutrophil-gene signal after depletion.
Output (results/r2_audit/): neutrophil_timer_replication.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

W = phase.W
blood = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_gene_table.csv").set_index("gene")["wt_zt13_vs_zt5_log2fc"]
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
neut = set(mk[mk["cell_type"].str.startswith("Neutrophil") & (mk["logfoldchanges"] >= 1.5) & (mk["pvals_adj"] < 1e-10)]["names"])
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
units = []
for f in sorted((RESULTS / "r2_external").glob("external_rhythm_GSE198770_*.csv")) + [RESULTS / "71_external/external_rhythm_GSE88954.csv"]:
    d = pd.read_csv(f, usecols=["dataset", "tissue", "condition", "gene", "a", "b", "p"])
    units.append(d)
d = pd.concat(units, ignore_index=True)
rows = []
for (ds, t, c), g in d.groupby(["dataset", "tissue", "condition"]):
    r = off[(off["dataset"] == ds) & (off["tissue"] == t) & (off["condition"] == c) & (off["unit"] == "bulk")]
    o = float(r["offset_canonical_h"].iloc[0]) if len(r) else 0.0
    t13, t5 = 13 + o, 5 + o                                   # internal ZT13 / ZT5 on the dataset's own axis
    g = g.drop_duplicates("gene").set_index("gene")
    pred = g["a"] * (np.cos(W * t13) - np.cos(W * t5)) + g["b"] * (np.sin(W * t13) - np.sin(W * t5))
    common = pred.index.intersection(blood.index)
    for subset, genes in [("all", common), ("neutrophil_markers", common.intersection(neut)),
                          ("rhythmic_p05", common.intersection(g.index[g["p"] < 0.05]))]:
        if len(genes) < 10:
            continue
        rho, p = stats.spearmanr(pred.loc[genes], blood.loc[genes])
        rows.append({"dataset": ds, "tissue": t, "condition": c, "genes": subset, "n_genes": len(genes),
                     "unit_offset_h": o, "spearman_rho": float(rho), "p": float(p)})
res = pd.DataFrame(rows)
ours = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_concordance.csv")
for r in ours.itertuples():
    res.loc[len(res)] = {"dataset": "this_atlas", "tissue": r.tissue, "condition": r.cell_type, "genes": "all (77)",
                         "n_genes": r.n_shared_genes, "unit_offset_h": np.nan, "spearman_rho": r.spearman_all, "p": r.p_perm_all}
write_csv(res, "r2_audit", "neutrophil_timer_replication.csv",
          "Correlation of predicted ZT13−ZT5 changes (external tissues, our cell types) with the blood-neutrophil change (GSE86619)",
          required=["dataset", "tissue", "condition", "genes", "spearman_rho", "p"])
log("\n" + res[res["genes"].isin(["all", "neutrophil_markers"])].round(3).to_string())
