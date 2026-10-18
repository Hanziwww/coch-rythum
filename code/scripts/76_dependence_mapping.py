"""Module 5.4f — glucocorticoid (GC)- and neutrophil-dependence of rhythmic genes, mapped onto our cell types.

External model selection (71_external, dryR): GSE88954 cochlea, sham vs adrenalectomy (ADX; constant darkness, CT)
and GSE198770 BM, control vs anti-Ly6G neutrophil depletion (LD, ZT). dryR's 5 models for two conditions are labelled
from which condition's rhythm parameters are zero in that model: arrhythmic / rhythmic only in condition X / same
rhythm / different rhythm. Dependence classes (control = sham or control):
  dependent    = rhythmic only in control (rhythm lost after ADX or neutrophil depletion)
  independent  = same rhythm in both conditions
  modulated    = rhythmic in both with different amplitude/phase
  gained       = rhythmic only after the intervention
  arrhythmic   = arrhythmic in both
'confident' calls require the dryR BIC weight of the chosen model >= 0.5.
Per tissue × cell type (our Tier-1/2 rhythmic genes vs our non-rhythmic tested genes, both restricted to genes
classified in the external dataset): count and fraction per class; one-sided Fisher exact test of enrichment of
rhythmic genes in each class (BH within dataset × tissue × class). Phase concordance per cell type: genes rhythmic in
our data (Tier 1/2) and in the external control condition (limma cosinor q < 0.05, 71_external) — circular
correlation (Jammalamadaka), mean offset (ours − external) and median absolute phase difference (CT taken as ZT).
Outputs (results/76_dependence/): dependence_gene_classes.csv, gc_dependence_mapping.csv,
neutrophil_dependence_mapping.csv, dependence_phase_concordance.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "76_dependence"
W = 2 * np.pi / 24
DATASETS = {"GSE88954": ("71_external/external_dryR_GSE88954_cochlea.csv", "sham", "adx", "gc"),
            "GSE198770": ("71_external/external_dryR_GSE198770_bone_marrow.csv", "control", "neutrophil_depleted",
                          "neutrophil")}


def label_models(d, ctrl, trt):
    """Map dryR model numbers to labels from which condition's amplitude is zero (majority over genes)."""
    lab = {}
    for k, g in d.groupby("chosen_model"):
        z_c = (g[f"amp_{ctrl}"].fillna(0).abs() < 1e-12).mean()      # dryR leaves NA for arrhythmic conditions
        z_t = (g[f"amp_{trt}"].fillna(0).abs() < 1e-12).mean()
        if z_c > 0.5 and z_t > 0.5:
            lab[k] = "arrhythmic"
        elif z_t > 0.5:
            lab[k] = "dependent"
        elif z_c > 0.5:
            lab[k] = "gained"
        else:
            same = (np.isclose(g[f"amp_{ctrl}"], g[f"amp_{trt}"]) & np.isclose(g[f"phase_{ctrl}"], g[f"phase_{trt}"]))
            lab[k] = "independent" if same.mean() > 0.5 else "modulated"
    return lab


def circ_corr(x, y):
    """Jammalamadaka–SenGupta circular correlation of two angle vectors (radians)."""
    mx, my = np.angle(np.exp(1j * x).mean()), np.angle(np.exp(1j * y).mean())
    sx, sy = np.sin(x - mx), np.sin(y - my)
    den = np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
    return (sx * sy).sum() / den if den > 0 else np.nan


cls_rows, map_rows, ph_rows = [], {"gc": [], "neutrophil": []}, []
ours = {t: pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv",
                       usecols=["group", "gene", "tier", "peak_zt"]) for t in ["cochlea", "bm"]}
for ds, (path, ctrl, trt, kind) in DATASETS.items():
    d = pd.read_csv(RESULTS / path)
    lab = label_models(d, ctrl, trt)
    log(f"{ds}: dryR model labels {lab}")
    d["class"] = d["chosen_model"].map(lab)
    d["confident"] = d["chosen_model_BICW"] >= 0.5
    ext = pd.read_csv(RESULTS / f"71_external/external_rhythm_{ds}.csv")
    ext = ext[ext["condition"] == ctrl].sort_values("p").drop_duplicates("gene").set_index("gene")
    cls_rows.append(d[["gene", "class", "chosen_model", "chosen_model_BICW", "confident"]].assign(
        dataset=ds, dependence=kind, control=ctrl, intervention=trt))
    dc = d.drop_duplicates("gene").set_index("gene")
    for tissue, t in ours.items():
        for ct, g in t.groupby("group"):
            g = g[g["gene"].isin(dc.index)]
            rhy = g["tier"].isin(["tier1", "tier2"]).to_numpy()
            if rhy.sum() < 10:
                continue
            for conf in [False, True]:
                cl = dc.loc[g["gene"], "class"].to_numpy()
                ok = dc.loc[g["gene"], "confident"].to_numpy() if conf else np.ones(len(g), bool)
                for c in ["dependent", "independent", "modulated", "gained", "arrhythmic"]:
                    inc = (cl == c) & ok
                    a, b = int((inc & rhy).sum()), int((~inc & rhy).sum())
                    cc, dd = int((inc & ~rhy).sum()), int((~inc & ~rhy).sum())
                    orr, p = stats.fisher_exact([[a, b], [cc, dd]], alternative="greater")
                    map_rows[kind].append({"dataset": ds, "tissue": tissue, "cell_type": ct, "class": c,
                                           "confident_only": conf, "n_rhythmic": int(rhy.sum()),
                                           "n_rhythmic_in_class": a, "frac_rhythmic_in_class": a / max(rhy.sum(), 1),
                                           "frac_nonrhythmic_in_class": cc / max((~rhy).sum(), 1),
                                           "odds_ratio": orr, "p_fisher_greater": p,
                                           "genes": ";".join(g.loc[inc & rhy, "gene"].head(200))})
            gg = g[rhy & g["gene"].isin(ext.index[ext["q_BH"] < 0.05])]
            if len(gg) >= 5:
                ours_h = gg["peak_zt"].to_numpy() % 24
                ext_h = ext.loc[gg["gene"], "peak_h"].to_numpy() % 24
                diff = (ours_h - ext_h + 12) % 24 - 12
                off = (np.angle(np.exp(1j * W * diff).mean()) / W)
                ph_rows.append({"dataset": ds, "tissue": tissue, "cell_type": ct, "n_genes": len(gg),
                                "circ_corr": circ_corr(W * ours_h, W * ext_h), "mean_offset_h": off,
                                "offset_resultant": float(np.abs(np.exp(1j * W * diff).mean())),
                                "median_abs_diff_h": float(np.median(np.abs(diff))),
                                "frac_within_3h": float((np.abs(diff) < 3).mean())})
write_csv(pd.concat(cls_rows, ignore_index=True), MOD, "dependence_gene_classes.csv",
          "dryR dependence class per gene (GC: GSE88954 sham vs ADX; neutrophil: GSE198770 control vs depletion)",
          required=["gene", "dataset", "class", "chosen_model_BICW"])
for kind, rows in map_rows.items():
    m = pd.DataFrame(rows)
    m["q_BH"] = m.groupby(["dataset", "tissue", "class", "confident_only"])["p_fisher_greater"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    write_csv(m, MOD, f"{kind}_dependence_mapping.csv",
              f"{'Glucocorticoid' if kind == 'gc' else 'Neutrophil'}-dependence classes among our rhythmic genes per "
              "cell type (Fisher enrichment vs non-rhythmic genes)",
              required=["dataset", "tissue", "cell_type", "class", "frac_rhythmic_in_class", "odds_ratio", "q_BH"])
write_csv(pd.DataFrame(ph_rows), MOD, "dependence_phase_concordance.csv",
          "Phase concordance of our rhythmic genes with the external control condition per cell type",
          required=["dataset", "tissue", "cell_type", "n_genes", "circ_corr", "mean_offset_h"])
log("done")
