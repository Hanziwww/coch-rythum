"""Round 4, WP-C — human relevance of the neutrophil rhythms: blood neutrophils of 3 healthy donors at 08:00 and 14:00
(GSE136632, normalised CPM, paired by donor).

Observed: per gene mean paired log2 difference 14:00 − 08:00 (log2 CPM + 1), paired t (n = 3).
Predicted: our neutrophil-state cosinor fits (33 tables, a / b on log2 scale; BM and cochlea; genes with p < 0.05)
give Δ = a(cos ωt₂ − cos ωt₁) + b(sin ωt₂ − sin ωt₁) at activity-matched mouse times. Humans wake ~07:00, i.e. the
start of the active phase = mouse ZT12, so 08:00 → ZT13 and 14:00 → ZT19 (primary); sensitivity shifts −2 … +2 h.
Mouse → human 1:1 orthologues (MGI HOM). Statistic: Pearson r between predicted and observed Δ across genes, gene-label
permutation p (2000 draws), sign agreement among genes with |observed| ≥ 0.3 and paired p < 0.1.
Outputs (results/r4_crossspecies/): human_neutrophil_timeofday.csv, human_neutrophil_gene_deltas.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_crossspecies"
W = 2 * np.pi / 24
rng = np.random.default_rng(11)
hom = pd.read_csv(EXTERNAL / "resources/mgi_hom_mouse_human.rpt", sep="\t", dtype=str)
hom = hom.rename(columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
mm = hom[hom["org"].str.startswith("mouse")].groupby("key")["symbol"].agg(list)
hs = hom[hom["org"].str.startswith("human")].groupby("key")["symbol"].agg(list)
m2h = {m[0]: h[0] for k in mm.index.intersection(hs.index) for h, m in [(hs[k], mm[k])] if len(h) == 1 and len(m) == 1}

h = pd.read_csv(EXTERNAL / "geo/GSE136632/GSE136632_normalized_cpm_humanRNAseq.csv.gz")
h = h.dropna(subset=["ext_gene"]).drop_duplicates("ext_gene").set_index("ext_gene")
donors = ["C", "E", "J"]
L = np.log2(h[[f"{d}_CT{t}" for d in donors for t in (8, 14)]].astype(float) + 1)
L = L[(h[[f"{d}_CT{t}" for d in donors for t in (8, 14)]].astype(float).mean(1) >= 5)]
dif = np.column_stack([L[f"{d}_CT14"] - L[f"{d}_CT8"] for d in donors])
obs = pd.DataFrame({"human_gene": L.index, "obs_delta": dif.mean(1),
                    "obs_p_paired": stats.ttest_1samp(dif, 0, axis=1).pvalue}).set_index("human_gene")
log(f"human neutrophils: {len(obs)} expressed genes")

rows, gene_rows = [], []
for t in ["bm", "cochlea"]:
    a = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "a", "b", "p", "status"])
    a = a[(a["status"] == "primary") & a["group"].str.startswith("Neutrophil") & (a["p"] < 0.05)]
    a["human_gene"] = a["gene"].map(m2h)
    a = a.dropna(subset=["human_gene"])
    for ct, g in list(a.groupby("group")) + [("all neutrophil states", a)]:
        g = g.sort_values("p").drop_duplicates("human_gene")
        g = g[g["human_gene"].isin(obs.index)]
        if len(g) < 20:
            continue
        o = obs.loc[g["human_gene"]]
        for shift in [-2, -1, 0, 1, 2]:
            t1, t2 = 13 + shift, 19 + shift
            pred = (g["a"] * (np.cos(W * t2) - np.cos(W * t1)) + g["b"] * (np.sin(W * t2) - np.sin(W * t1))).to_numpy()
            ov = o["obs_delta"].to_numpy()
            r = float(np.corrcoef(pred, ov)[0, 1])
            null = np.array([np.corrcoef(pred, rng.permutation(ov))[0, 1] for _ in range(2000)])
            strong = (np.abs(ov) >= 0.3) & (o["obs_p_paired"].to_numpy() < 0.1)
            rows.append({"tissue": t, "cell_type": ct, "mapping": f"08:00→ZT{t1 % 24}, 14:00→ZT{t2 % 24}", "shift_h": shift,
                         "primary": shift == 0, "n_genes": len(g), "pearson_r": r, "p_perm": float((1 + (null >= r).sum()) / 2001),
                         "n_strong_human_changes": int(strong.sum()),
                         "sign_agreement_strong": float((np.sign(pred[strong]) == np.sign(ov[strong])).mean()) if strong.sum() else np.nan})
            if shift == 0 and ct == "all neutrophil states":
                gene_rows.append(g.assign(tissue=t, predicted_delta=pred, observed_delta=ov, observed_p_paired=o["obs_p_paired"].to_numpy())
                                 [["tissue", "group", "gene", "human_gene", "p", "predicted_delta", "observed_delta", "observed_p_paired"]])
res = pd.DataFrame(rows)
write_csv(res, MOD, "human_neutrophil_timeofday.csv",
          "Human blood neutrophils 14:00 vs 08:00 (GSE136632) vs the difference predicted by our neutrophil-state rhythms at activity-matched mouse times",
          required=["tissue", "cell_type", "mapping", "n_genes", "pearson_r", "p_perm"])
write_csv(pd.concat(gene_rows, ignore_index=True), MOD, "human_neutrophil_gene_deltas.csv",
          "Per gene: predicted (our neutrophil-state cosinor, ZT19 − ZT13) and observed human 14:00 − 08:00 log2 differences",
          required=["tissue", "gene", "human_gene", "predicted_delta", "observed_delta"])
log("\n" + res[res["primary"]].round(3).to_string())
log("\n" + res[res["cell_type"] == "all neutrophil states"].round(3).to_string())
